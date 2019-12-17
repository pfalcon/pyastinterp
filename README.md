pyastinterp
===========

This is an AST interpreter (also known as tree-walking interpreter) for
the Python programming language, written in Python itself. It is thus
a meta-circular interpreter. While Python implementations of a Python VM
(i.e. bytecode interpreters) are relatively well known (starting with
PyPy), AST interpreters are much less common beyond those which implement
a very limited subset of the syntax.

Pyastinterp strives to fulfil 2 goals:

1. Be as complete as possible (and see how much effort that takes and
   if there're any limits).
2. Yet be as minimal as possible, avoiding useless boilerplate and
   reimplementing what may be already available.

First goal is pretty obvious: ideally, we'd like Pyastinterp to be able
to interpret (run) any application a native Python interpreter can.

Second goal is rooted in many reasons. To start with, Pyastinterp is
a spiritual part of the [Pycopy](https://github.com/pfalcon/pycopy)
project, whose goal is the development of minimalist Python-compatible
ecosystem. Besides, "bloated" meta-circular interpreters are well out
there (the same PyPy) - there's lack on the opposite side of spectrum.
Pyastinterp is also intended to be a "teaching" and "for-research"
project. Surely, it's much easier to learn and reuse/repurpose a small,
rather than large, project. These last requirements however also put
bounds on "minimality": `pyastinterp` tries to have a clear focus
(AST interpretation) and avoid unneeded boilerplate, but not be
obfuscated, and actually tries to be clear and easily hackable (provide
some reusable infrastructure), even at the expense of somewhat less
optimality/performance.

To achieve the goals of minimality, `pyastinterp` does following:

1. Is concerned with syntax, not semantics (largely).
2. That means that it tries to interpret Python AST tree, but
   relies on the underlying Python runtime to implement the actual
   operations (semantics). In some cases `pyastinterp` actually
   has to deal with runtime semantics, but the point is that it
   should be done only when there's no other choice (i.e. rarely).
3. This in particular means that `pyastinterp` itself requires a
   (pretty) full Python implementation to run, it's not the case
   that it's written in "some subset of Python". (Contrast that
   with e.g. PyPy's RPython, which is a Python subset language,
   in which interpreters are written).
4. Another consequence is that there's no semantic separation
   between "metalevel" (level of the interpreter) and target
   application level. This is again unlike PyPy, where 2 are explicitly
   separated. Lack of separation allows to save on "marshalling"
   values between the 2 levels, but also make it possible to have
   "leaks" between levels, e.g. unexpected and unhandled exception
   in the interpreter to be delivered to the application, causing
   havoc and confusion. Pyastinterp's response to this concern
   is a pledge that there shouldn't be such unhandled interpreter-level
   exceptions. Of course, catching all such cases may take quite
   a lot of testing and debugging, and fixing them may affect code
   minimality and clarity. We'll see how it goes.
5. However, due to choices described above, implementation of many
   syntactic constructs is very clear, concise, and isomorphic
   between the target and meta levels. See e.g. how "if" statement
   is implemented: the implementation looks almost exactly how the
   usage of the "if" statement is.


Credits and licensing
---------------------

Pyastinterp is (c) [Paul Sokolovsky](https://github.com/pfalcon) and
is released under the MIT license.
