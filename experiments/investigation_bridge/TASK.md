# Investigation bridge: affine frame repair

This public synthetic workload measures whether several investigators improve a
bounded code repair over a solo attempt. It was written for this experiment and
does not contain fixtures, code, or incidents from a private or work repository.

The candidate receives the public contract, one faulty standalone `solution.py`,
and 13 development requests. It repairs `evaluate(request)`, which converts an
exact rational 2D point between frames in a rooted affine graph. The evaluator
runs 52 fresh final requests after the repair is frozen. Frame names are opaque,
and the final split varies ancestor depth and branch topology rather than replaying
renamed development shapes.

`workload.task(seed, variant)` returns:

- `id`: stable task identity for that seed and variant;
- `contract`: the complete public behavior and error precedence;
- `starter`: standalone Python source defining `evaluate(request)`;
- `development`: requests available during investigation;
- `final`: requests reserved for the evaluation harness.

The six single-fault variants are `composition`, `inverse_translation`, `pivot`,
`singular_conversion`, `ancestor_omission`, and `sibling_paths`. `hard` combines
all six. The source uses only the Python standard library.

The model must not receive `workload.reference`, `workload.correct_source()`, or
the `final` requests. The harness may use them after candidate work ends.
`reference(request)` is an exact `Fraction` oracle built with homogeneous
matrices. The known-good candidate intentionally uses separate scalar path
walking. `mutants()` exposes the six single-fault starters plus a regression
control that checks only the target frame's local determinant instead of its
whole path to world.

Run the workload checks from the repository root:

```sh
python3 -m unittest discover -s experiments/investigation_bridge -p 'test_workload.py' -v
```
