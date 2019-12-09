# Python AST interpreter written in Python
#
# This module is part of the Pycopy https://github.com/pfalcon/pycopy
# project.
#
# Copyright (c) 2019 Paul Sokolovsky
#
# The MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import ast
import builtins


class StrictNodeVisitor(ast.NodeVisitor):

    def generic_visit(self, node):
        n = node.__class__.__name__
        raise NotImplementedError("Visitor for node {} not implemented".format(n))


# Pycopy by default doesn't support direct slice construction, use helper
# object to construct it.
class SliceGetter:

    def __getitem__(self, idx):
        return idx

slice_getter = SliceGetter()


class TargetNonlocalFlow(Exception):
    """Base exception class to simulate non-local control flow transfers in
    a target application."""
    pass

class TargetBreak(TargetNonlocalFlow):
    pass

class TargetContinue(TargetNonlocalFlow):
    pass

class TargetReturn(TargetNonlocalFlow):
    pass


class VarScopeSentinel:

    def __init__(self, name):
        self.name = name

NO_VAR = VarScopeSentinel("no_var")
GLOBAL = VarScopeSentinel("global")


class InterpFunc:
    "Callable wrapper for AST functions (FunctionDef nodes)."

    def __init__(self, node, interp):
        self.node = node
        self.interp = interp

    def __call__(self, *args, **kwargs):
        return self.interp.call_func(self.node, *args, **kwargs)


class Interpreter(StrictNodeVisitor):

    def __init__(self):
        self.ns = {}
        self.ns_stack = []
        # To implement "store" operation, we need to arguments: location and
        # value to store. The operation itself is handled by a node visitor
        # (e.g. visit_Name), and location is represented by AST node, but
        # there's no support to pass additional arguments to a visitor
        # (likely, because it would be a burden to explicit pass such
        # additional arguments thru the chain of visitors). So instead, we
        # store this value as field. As interpretation happens sequentially,
        # there's no risk that it will be overwritten "concurrently".
        self.store_val = None
        # Current active exception, for bare "raise", which doesn't work
        # across function boundaries (and that's how we have it - exception
        # would be caught in visit_Try, while re-rasing would happen in
        # visit_Raise).
        self.cur_exc = None

    def stmt_list_visit(self, lst):
        res = None
        for s in lst:
            res = self.visit(s)
        return res

    def visit_Module(self, node):
        self.stmt_list_visit(node.body)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_FunctionDef(self, node):
        # Defaults are evaluated at function definition time, so we
        # need to do that now.
        self.prepare_func(node)
        self.ns[node.name] = InterpFunc(node, self)

    def prepare_func(self, node):
        """Prepare function AST node for future interpretation: pre-calculate
        and cache useful information, etc."""
        args = node.args
        num_required = len(args.args) - len(args.defaults)
        all_args = set()
        d = {}
        for i, a in enumerate(args.args):
            all_args.add(a.arg)
            if i >= num_required:
                d[a.arg] = self.visit(args.defaults[i - num_required])
        for a, v in zip(args.kwonlyargs, args.kw_defaults):
            all_args.add(a.arg)
            if v is not None:
                d[a.arg] = self.visit(v)
        node.args.defaults_dict = d
        node.args.all_args = all_args

    def prepare_func_args(self, node, *args, **kwargs):

        def arg_num_mismatch():
            raise TypeError("{}() takes {} positional arguments but {} were given".format(node.name, len(argspec.args), len(args)))

        argspec = node.args

        # If there's vararg, either offload surplus of args to it, or init
        # it to empty tuple (all in one statement). If no vararg, error on
        # too many args.
        if argspec.vararg:
            self.ns[argspec.vararg.arg] = args[len(argspec.args):]
        else:
            if len(args) > len(argspec.args):
                arg_num_mismatch()

        for i in range(min(len(args), len(argspec.args))):
            self.ns[argspec.args[i].arg] = args[i]

        # Process incoming keyword arguments, putting them in namespace if
        # actual arg exists by that name, or offload to function's kwarg
        # if any. All make needed checks and error out.
        func_kwarg = {}
        for k, v in kwargs.items():
            if k in argspec.all_args:
                if k in self.ns:
                    raise TypeError("{}() got multiple values for argument '{}'".format(node.name, k))
                self.ns[k] = v
            elif argspec.kwarg:
                func_kwarg[k] = v
            else:
                raise TypeError("{}() got an unexpected keyword argument '{}'".format(node.name, k))
        if argspec.kwarg:
            self.ns[argspec.kwarg.arg] = func_kwarg

        # Finally, overlay default values for arguments not yet initialized.
        # We need to do this last for "multiple values for the same arg"
        # check to work.
        for k, v in argspec.defaults_dict.items():
            if k not in self.ns:
                self.ns[k] = v

        # And now go thru and check for any missing arguments.
        for a in argspec.args:
            if a.arg not in self.ns:
                raise TypeError("{}() missing required positional argument: '{}'".format(node.name, a.arg))
        for a in argspec.kwonlyargs:
            if a.arg not in self.ns:
                raise TypeError("{}() missing required keyword-only argument: '{}'".format(node.name, a.arg))

    def call_func(self, node, *args, **kwargs):
        self.ns_stack.append(self.ns)
        self.ns = {}
        try:
            self.prepare_func_args(node, *args, **kwargs)
            res = self.stmt_list_visit(node.body)
        except TargetReturn as e:
            res = e.args[0]
        finally:
            self.ns = self.ns_stack.pop()
        return res

    def visit_Return(self, node):
        if not self.ns_stack:
            raise SyntaxError("'return' outside function")
        raise TargetReturn(node.value and self.visit(node.value))

    def visit_Try(self, node):
        try:
            self.stmt_list_visit(node.body)
        except TargetNonlocalFlow:
            raise
        except Exception as e:
            self.cur_exc = e
            for h in node.handlers:
                if h.type is None or isinstance(e, self.visit(h.type)):
                    if h.name:
                        self.ns[h.name] = e
                    self.stmt_list_visit(h.body)
                    if h.name:
                        del self.ns[h.name]
                    self.cur_exc = None
                    break
            else:
                raise
        else:
            self.stmt_list_visit(node.orelse)
        finally:
            self.stmt_list_visit(node.finalbody)
        # Could use "finally:" here to not repeat
        # stmt_list_visit(node.finalbody) 3 times

    def visit_For(self, node):
        iter = self.visit(node.iter)
        for item in iter:
            self.store_val = item
            self.visit(node.target)
            try:
                self.stmt_list_visit(node.body)
            except TargetBreak:
                break
            except TargetContinue:
                continue
        else:
            self.stmt_list_visit(node.orelse)

    def visit_While(self, node):
        while self.visit(node.test):
            try:
                self.stmt_list_visit(node.body)
            except TargetBreak:
                break
            except TargetContinue:
                continue
        else:
            self.stmt_list_visit(node.orelse)

    def visit_Break(self, node):
        raise TargetBreak

    def visit_Continue(self, node):
        raise TargetContinue

    def visit_If(self, node):
        test = self.visit(node.test)
        if test:
            self.stmt_list_visit(node.body)
        else:
            self.stmt_list_visit(node.orelse)

    def visit_Import(self, node):
        for n in node.names:
            self.ns[n.asname or n.name] = __import__(n.name)

    def visit_ImportFrom(self, node):
        mod = __import__(node.module, None, None, [n.name for n in node.names], node.level)
        for n in node.names:
            self.ns[n.asname or n.name] = getattr(mod, n.name)

    def visit_Raise(self, node):
        if node.exc is None:
            raise self.cur_exc
        if node.cause is None:
            raise self.visit(node.exc)
        else:
            raise self.visit(node.exc) from self.visit(node.cause)

    def visit_AugAssign(self, node):
        assert isinstance(node.target.ctx, ast.Store)
        # Not functional style, oops. Node in AST has store context, but we
        # need to read its value first. To not construct a copy of the entire
        # node with load context, we temporarily patch it in-place.
        save_ctx = node.target.ctx
        node.target.ctx = ast.Load()
        var_val = self.visit(node.target)
        node.target.ctx = save_ctx

        rval = self.visit(node.value)

        # As augmented assignment is statement, not operator, we can't put them
        # all into map. We could instead directly lookup special inplace methods
        # (__iadd__ and friends) and use them, with a fallback to normal binary
        # operations, but from the point of view of this interpreter, presence
        # of such methods is an implementation detail of the object system, it's
        # not concerned with it.
        op = type(node.op)
        if op is ast.Add:
            var_val += rval
        elif op is ast.Sub:
            var_val -= rval
        elif op is ast.Mult:
            var_val *= rval
        elif op is ast.Div:
            var_val /= rval
        elif op is ast.FloorDiv:
            var_val //= rval
        elif op is ast.Mod:
            var_val %= rval
        elif op is ast.Pow:
            var_val **= rval
        elif op is ast.LShift:
            var_val <<= rval
        elif op is ast.RShift:
            var_val >>= rval
        elif op is ast.BitAnd:
            var_val &= rval
        elif op is ast.BitOr:
            var_val |= rval
        elif op is ast.BitXor:
            var_val ^= rval
        else:
            raise NotImplementedError

        self.store_val = var_val
        self.visit(node.target)

    def visit_Assign(self, node):
        self.store_val = self.visit(node.value)
        for n in node.targets:
            self.visit(n)

    def visit_Pass(self, node):
        pass

    def visit_Expr(self, node):
        # Produced value is ignored
        self.visit(node.value)

    def visit_Call(self, node):
        func = self.visit(node.func)
        args = [self.visit(a) for a in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        return func(*args, **kwargs)

    def visit_Compare(self, node):
        cmpop_map = {
            ast.Eq: lambda x, y: x == y,
            ast.NotEq: lambda x, y: x != y,
            ast.Lt: lambda x, y: x < y,
            ast.LtE: lambda x, y: x <= y,
            ast.Gt: lambda x, y: x > y,
            ast.GtE: lambda x, y: x >= y,
            ast.Is: lambda x, y: x is y,
            ast.IsNot: lambda x, y: x is not y,
            ast.In: lambda x, y: x in y,
            ast.NotIn: lambda x, y: x not in y,
        }
        lv = self.visit(node.left)
        for op, r in zip(node.ops, node.comparators):
            rv = self.visit(r)
            if not cmpop_map[type(op)](lv, rv):
                return False
            lv = rv
        return True

    def visit_BoolOp(self, node):
        if isinstance(node.op, ast.And):
            res = True
            for v in node.values:
                res = res and self.visit(v)
        elif isinstance(node.op, ast.Or):
            res = False
            for v in node.values:
                res = res or self.visit(v)
        else:
            raise NotImplementedError
        return res

    def visit_BinOp(self, node):
        binop_map = {
            ast.Add: lambda x, y: x + y,
            ast.Sub: lambda x, y: x - y,
            ast.Mult: lambda x, y: x * y,
            ast.Div: lambda x, y: x / y,
            ast.FloorDiv: lambda x, y: x // y,
            ast.Mod: lambda x, y: x % y,
            ast.Pow: lambda x, y: x ** y,
            ast.LShift: lambda x, y: x << y,
            ast.RShift: lambda x, y: x >> y,
            ast.BitAnd: lambda x, y: x & y,
            ast.BitOr: lambda x, y: x | y,
            ast.BitXor: lambda x, y: x ^ y,
        }
        l = self.visit(node.left)
        r = self.visit(node.right)
        return binop_map[type(node.op)](l, r)

    def visit_UnaryOp(self, node):
        unop_map = {
            ast.UAdd: lambda x: +x,
            ast.USub: lambda x: -x,
            ast.Invert: lambda x: ~x,
            ast.Not: lambda x: not x,
        }
        val = self.visit(node.operand)
        return unop_map[type(node.op)](val)

    def visit_Subscript(self, node):
        obj = self.visit(node.value)
        idx = self.visit(node.slice)
        if isinstance(node.ctx, ast.Load):
            return obj[idx]
        elif isinstance(node.ctx, ast.Store):
            obj[idx] = self.store_val
        elif isinstance(node.ctx, ast.Del):
            del obj[idx]
        else:
            raise NotImplementedError

    def visit_Index(self, node):
        return self.visit(node.value)

    def visit_Slice(self, node):
        # Any of these can be None
        lower = node.lower and self.visit(node.lower)
        upper = node.upper and self.visit(node.upper)
        step = node.step and self.visit(node.step)
        slice = slice_getter[lower:upper:step]
        return slice

    def visit_Attribute(self, node):
        obj = self.visit(node.value)
        if isinstance(node.ctx, ast.Load):
            return getattr(obj, node.attr)
        elif isinstance(node.ctx, ast.Store):
            setattr(obj, node.attr, self.store_val)
        elif isinstance(node.ctx, ast.Del):
            delattr(obj, node.attr)
        else:
            raise NotImplementedError

    def visit_Global(self, node):
        for n in node.names:
            if n in self.ns and self.ns[n] is not GLOBAL:
                raise SyntaxError("SyntaxError: name '{}' is assigned to before global declaration".format(n))
            if self.ns_stack:
                self.ns[n] = GLOBAL

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            res = NO_VAR
            if node.id in self.ns:
                res = self.ns[node.id]
            else:
                for ns in reversed(self.ns_stack):
                    if node.id in ns:
                        res = ns[node.id]
                        break

            if res is GLOBAL:
                res = self.ns_stack[0].get(node.id, NO_VAR)
            if res is not NO_VAR:
                return res

            try:
                return getattr(builtins, node.id)
            except AttributeError:
                raise NameError("name '{}' is not defined".format(node.id))
        elif isinstance(node.ctx, ast.Store):
            res = self.ns.get(node.id, NO_VAR)
            if res is GLOBAL:
                self.ns_stack[0][node.id] = self.store_val
            else:
                self.ns[node.id] = self.store_val
        elif isinstance(node.ctx, ast.Del):
            res = self.ns.get(node.id, NO_VAR)
            if res is NO_VAR:
                raise NameError("name '{}' is not defined".format(node.id))
            elif res is GLOBAL:
                del self.ns_stack[0][node.id]
            else:
                del self.ns[node.id]
        else:
            raise NotImplementedError

    def visit_Dict(self, node):
        return {self.visit(p[0]): self.visit(p[1]) for p in zip(node.keys, node.values)}

    def visit_List(self, node):
        return [self.visit(e) for e in node.elts]

    def visit_Tuple(self, node):
        return tuple([self.visit(e) for e in node.elts])

    def visit_NameConstant(self, node):
        return node.value

    def visit_Ellipsis(self, node):
        return ...

    def visit_Str(self, node):
        return node.s

    def visit_Bytes(self, node):
        return node.s

    def visit_Num(self, node):
        return node.n


if __name__ == "__main__":
    import sys
    tree = ast.parse(open(sys.argv[1]).read())
    #print(ast.dump(tree))

    interp = Interpreter()
    interp.visit(tree)
