# Validation record

The POC was exercised locally on 2026-09-12. Numerical outcomes are reported separately in EXERIMENT.md; software checks do not establish model usefulness.

## Automated checks

`python -m pytest -q`: 32 passing tests at the current implementation stage. Checks cover:

- seed provenance and exact cross-split input separation;
- malformed model outputs, empty/invented quotes, invalid source-line selections;
- separate label accuracy, schema validity, and markdown-wrapped JSON handling;
- inference messages excluding example datasets and external labels;
- importing a second skill and rejecting duplicate identities;
- persisted corrections, replacement semantics, and protection of test/validation cases;
- unavailable model/reference handling and exclusive runtime admission;
- cross-origin writes, host validation, and interrupted run recovery.

`node --check static/app.js`: passed.

## Real execution

- Installed MLX-LM locally and downloaded hash-pinned model revisions.
- Trained real QLoRA adapters; preserved weights hashes, training telemetry, and recipes.
- Evaluated actual raw model responses against the 51 copied Workbench cases.
- Kept the initial unsuccessful experiments; no scores were fabricated or substituted from a different model.

## Browser checks

Using a real browser against the local HTTP server:

- Loaded the workshop and active CI skill from its API.
- Imported the support-routing template through the UI, then selected it.
- Saved a new support-routing correction; the practice count increased from 4 to 5.
- Opened experiment details, the measured loss plot, raw logs, and individual test cases.
- Inspected a complete 51-case result table and its source-log/response detail.
- Checked 1440-pixel desktop and 390-pixel mobile widths without horizontal overflow.

No cloud reference provider or paid training service was configured. No Workbench source, delivery state, live CI, or merge authority was changed.
