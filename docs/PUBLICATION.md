# Publication notes

Specialist Workshop is a local research application with inspected examples and retained experimental results. Public source does not make it a hosted service or qualify any of its trained models for production.

## Included material

- The application, tests, synthetic task packages, and MIT license.
- The copied CI evaluation corpus with its original license, revision, and byte hash. These are personal-project CI excerpts and historical labels, not independently adjudicated diagnoses.
- Synthetic analytics and workflow examples, plus explicitly identified retained personal experiment cases.
- Original model outputs, failed runs, and provenance needed to interpret the recorded results.

Historical logs and provenance can contain the author's original workstation paths. They are inert experiment metadata, not runtime configuration. They remain unchanged so the recorded hashes and raw evidence keep their meaning. The seeding script reads the bundled licensed corpus; the optional retained-case replay requires an explicit source-file argument.

Model weights, adapters, credentials, imported user data, local state, and books are not part of the repository. The design notes discuss ideas from a book; its text and code are not redistributed.

## First-run check

The README includes a deterministic `majority` evaluation so a new user can import a package and inspect a real recorded run without an API key or model download. It is labeled as a control. Model inference and training remain separate, explicit operations on Apple Silicon.

The release checks include the full Python test suite, all four browser-script syntax checks, the documented starter import and majority evaluation in fresh state, inspected desktop/mobile browser captures, and a secret scan over Git history. Numerical training results and their limits remain in [READINESS.md](READINESS.md).
