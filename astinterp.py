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


class Interpreter(StrictNodeVisitor):

    def __init__(self):
        self.ns = {}
        # To implement "store" operation, we need to arguments: location and
        # value to store. The operation itself is handled by a node visitor
        # (e.g. visit_Name), and location is represented by AST node, but
        # there's no support to pass additional arguments to a visitor
        # (likely, because it would be a burden to explicit pass such
        # additional arguments thru the chain of visitors). So instead, we
        # store this value as field. As interpretation happens sequentially,
        # there's no risk that it will be overwritten "concurrently".
        self.store_val = None

    def stmt_list_visit(self, lst):
        for s in lst:
            self.visit(s)

    def visit_Module(self, node):
        self.stmt_list_visit(node.body)

    def visit_Expression(self, node):
        return self.visit(node.body)

    def visit_Try(self, node):
        try:
            self.stmt_list_visit(node.body)
            self.stmt_list_visit(node.finalbody)
        except Exception as e:
            for h in node.handlers:
                if h.type is None or isinstance(e, self.visit(h.type)):
                    if h.name:
                        self.ns[h.name] = e
                    self.stmt_list_visit(h.body)
                    if h.name:
                        del self.ns[h.name]
                    self.stmt_list_visit(node.finalbody)
                    break
            else:
                self.stmt_list_visit(node.finalbody)
                raise
        # Could use "finally:" here to not repeat
        # stmt_list_visit(node.finalbody) 3 times

    def visit_For(self, node):
        iter = self.visit(node.iter)
        for item in iter:
            self.store_val = item
            self.visit(node.target)
            self.stmt_list_visit(node.body)
        else:
            self.stmt_list_visit(node.orelse)

    def visit_If(self, node):
        test = self.visit(node.test)
        if test:
            self.stmt_list_visit(node.body)
        else:
            self.stmt_list_visit(node.orelse)

    def visit_Import(self, node):
        for n in node.names:
            self.ns[n.asname or n.name] = __import__(n.name)

    def visit_Raise(self, node):
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

    def visit_Expr(self, node):
        # Produced value is ignored
        self.visit(node.value)

    def visit_Call(self, node):
        func = self.visit(node.func)
        args = [self.visit(a) for a in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        return func(*args, **kwargs)

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

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            if node.id in self.ns:
                return self.ns[node.id]
            try:
                return getattr(builtins, node.id)
            except AttributeError:
                raise NameError("name '{}' is not defined".format(node.id))
        elif isinstance(node.ctx, ast.Store):
            self.ns[node.id] = self.store_val
        elif isinstance(node.ctx, ast.Del):
            del self.ns[node.id]
        else:
            raise NotImplementedError

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

    def visit_Num(self, node):
        return node.n


if __name__ == "__main__":
    import sys
    tree = ast.parse(open(sys.argv[1]).read())
    #print(ast.dump(tree))

    interp = Interpreter()
    interp.visit(tree)
