# Specialist Workshop

Read README.md and docs/DESIGN.md. This is a local product POC, separate from Workbench's production tools.

- Preserve real execution. Never replace model training or inference with fabricated scores, mocked predictions, or retrieval labeled as fine-tuning.
- Keep external evaluation inputs out of training. Preserve snapshots, raw outputs, and exact model/checkpoint identities.
- A valid quote is not a correct diagnosis. Historical labels and synthetic data are explicitly bounded evidence.
- Skill-specific policy belongs in skill packages; numerical model execution belongs in the backend.
- Bind to localhost and run one app worker. Do not expose this POC as a hosted service without a separate design.
- Keep `.state`, credentials, model weights, and the user's books out of Git.
- Tests: `.venv/bin/python -m pytest -q`, `node --check static/app.js`, then exercise changed browser flows.
- Do not mutate Workbench or its authority state. Do not merge without the operator's authority and the applicable gate flow.
