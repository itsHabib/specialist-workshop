# Rectangle-union work-loop fixture

`taskpack.py` generates two related repair tasks and keeps final inputs and the
exact oracle outside the worker-visible `files` mapping. Variant 0 exercises the
library and stdin CLI. Variant 1 adds comments, file input, and compact JSON.

The evaluator accepts the frozen run-plan `variant` explicitly, calls
`execute(files, command, stdin="")`, and expects `stdout`,
`stderr`, `returncode`, `error`, `seconds`, and `cleanup_confirmed`. This is
deliberately transport-neutral; the experiment runner executes it in a fresh,
networkless container. `development_probe` provides bounded counterexamples from
a distribution and seed distinct from final evaluation.

The starter uses floats and pairwise overlap subtraction. It therefore fails on
large exact coordinates, reversed rectangles, and three-way overlap while still
being coherent enough for iterative diagnosis and repair.
