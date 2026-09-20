# First capability-policy runner

Built a working comparison path over existing Workshop task packages: iterative solo,
fresh-context builder–critic, train-only practice retrieval and stronger-model critique.
OpenAI and Anthropic text adapters share the same engine. No weights changed.

## What passed

- 137 tests across Workshop, including 41 targeted engine/provider/CLI tests. Test transports
  are deliberately fake and establish mechanics only, not model quality.
- Dry-run manifests for support intake and analytics planning; no credentials or state writes.
- Chromium opened the real failed-run report at 390px; no page errors or document overflow,
  failure status and unknown-cost accounting visible.
- Independent review verified shared budgets, absence of final-answer feedback, and repairs
  for interrupted spend summaries, missing cache/response provenance and omitted scorer hashes.
- Additional review probes confirmed a dollar reservation can stop the first or subsequent
  call before transport. Supplied rates remain assumptions, not invoice verification.

## What happened live

One request using the configured OpenAI credential returned HTTP401. The runner stopped,
recorded the attempted request and retained the $0.0832875 reservation; actual cost remains
unknown. That reservation is not money known to have been spent. No successful model
response or policy comparison occurred. The [smoke receipt](live-smoke.json) records it.
Raw prompt events and the report remain under ignored `.state/capability-smoke/`.
No Anthropic credential was present, so its adapter was validated with transport tests only.

## What this does not prove

No claim of Opus/Sol/Luna matching Astra, cost savings, hard-algorithm reasoning or improvement
on unseen tasks. Existing starter examples are exposed diagnostics. This slice orchestrates
text model calls and installed validators, not arbitrary tools, search, shell execution,
parallel hypotheses or Fleet workers. Development feedback is separated from final labels;
a human repeatedly examining final reports can still overfit the evaluation.

Next: restore a valid API credential and run a tiny complete diagnostic. Then choose a real
failure-prone task with fresh, adjudicated cases and a stronger evaluator where appropriate.
Add tools/decomposition only when the observed failure suggests they can help, and compare
against solo iteration at measured total cost. Preserve negative results.
