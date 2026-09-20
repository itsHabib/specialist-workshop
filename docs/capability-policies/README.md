# Make the whole system better than the model alone

Compare complete solving policies: iterative solo, independent critique, retrieved practice
examples, and selective stronger-model critique. The useful question is whether a setup
finishes your tasks reliably at an acceptable total cost—not whether two agents agree.

This first slice reuses Workshop task packages. It supports structured reasoning tasks
such as support diagnosis, analytics planning and CI evidence interpretation. It is not
restricted to algorithms, and it does not yet run arbitrary code, browser tools or repository
agents. Existing package validators supply deployable feedback; final answers never do.

## Preview a comparison without spending

From the repo root using the existing environment:

```sh
.venv/bin/python -m workshop.compare_policies \
  --package examples/support-intake.package.json \
  --config examples/capability-policies.json \
  --output .state/my-policy-run --dry-run
```

Remove `--dry-run` to run actual calls. Set `OPENAI_API_KEY` for OpenAI models and
`ANTHROPIC_API_KEY` for Anthropic models. Output directories must be new: uncertain
requests are never silently resumed or reissued. The run writes `run.json`, fsynced
before/after-call `events.jsonl`, a package snapshot and a self-contained `report.html`.
A dry run writes no files and needs no credentials.

Try `examples/roxiq-planning.package.json` or `examples/ci-practice.package.json` for
other task shapes. Starter cases are illustrative and exposed; results are diagnostic,
not held-out qualification. Bring a fresh, adjudicated package for a capability claim.

## What you can vary

The sample config compares five policies under a common per-task call/token ceiling:

| Policy | Change being tested |
|---|---|
| Cheaper solo | Iterative self-correction with the installed validator |
| Cheaper pair | Fresh critic context, then solver revision |
| Cheaper with practice | Fixed lexical retrieval from training examples only |
| Cheaper + stronger critique | Stronger model critiques; cheaper model owns the answer |
| Stronger solo | Same tools and iterative opportunity, no one-shot handicap |

`rounds` counts solver candidates. A pair with two rounds spends solver → critic → solver,
three calls total. All roles consume the same `max_calls`/`max_total_tokens` limits.
`max_output_tokens` bounds each completion. Early limits are recorded. The last solver
candidate is selected, even if a previous candidate would score better; final labels
never select the winner. A schema-valid answer may still be wrong, so configured refinement
continues instead of treating formatting success as semantic success.

Use an explicit Anthropic alias to test your available Opus model:

```json
{"provider":"anthropic","model":"YOUR_AVAILABLE_MODEL_ID"}
```

Change a policy's `solver` or `critic` alias to select it. Anthropic effort controls are
not implemented in this first adapter; unsupported options fail before paid requests.
OpenAI aliases support the configured `effort`. Receipts retain requested and returned
model identities. These model calls have no filesystem, shell or browser access.

## Costs and comparison integrity

The sample has a $2 whole-run reservation and 30-call maximum. Reservations use a
conservative text-input byte bound and maximum output, with the highest configured input
rate; reservations are retained even after failed requests. All model aliases require
explicit rates when a dollar limit is configured. Without rates costs remain unknown.
Never infer that equal tokens or calls mean equal dollars across models.

The sample standard-rate estimates were checked on 2026-09-20 against the official
[Luna](https://developers.openai.com/api/docs/models/gpt-5.6-luna) and
[Astra](https://developers.openai.com/api/docs/models/gpt-6-astra) pages. Refresh rates when
changing models, service tiers or prices; these are usage estimates, not reconciled invoices.
Priced calls above 128,000 estimated input tokens are refused rather than using a wrong
long-context tier. This implementation uses plain-text requests, not paid tools.

`known_cost_usd` retains known spend when an interrupted or usage-less call makes the total
unknown. Whole-run accounting includes paid calls from incomplete trajectories. Missing
usage stops the run. Transport failures cannot become accepted answers. Every request is
checkpointed before sending, so a process death can leave a visible uncertain call.

Randomized task/policy order and a shared config make repeatable comparisons possible;
this is not yet a statistical benchmark. Incomplete plans are marked; report tables contain
attempted rows only. Compare completed matched cohorts, total cost per successful task,
and false acceptance before making a claim. Critic quality and validity are not synonymous.

Training retrieval excludes development/final examples. Package/scorer/provider code hashes,
selected practice IDs, candidate hashes, response IDs and numeric cache/reasoning usage are
retained. Gold scoring happens after the policy terminates and is not sent to a model.
Final split evaluation is explicit (`--split final`); evaluating a set repeatedly exposes it
to the operator, so reserve new tasks after tuning. The runner does not enforce human blinding.

## What comes next

Add one real task where the weaker model fails, then test which intervention actually
helps: better context, task-specific instruments, decomposition, parallel hypotheses or
critic feedback. Add each mechanism independently enough to attribute the improvement.
This runner's `complete(model, messages, max_output_tokens)` seam can later be supplied by
an agent executor; shell/tool isolation and tool-cost accounting must be established first.
Fleet can own job execution and recovery without owning evaluation answers or task semantics.

A useful result may be that plain solo iteration wins. Keep the successful policy and
counterexamples; do not add coordination just to make the architecture look impressive.

Provider contracts: [OpenAI Responses](https://developers.openai.com/api/reference/typescript/resources/beta/subresources/responses/methods/create),
[Anthropic Messages](https://platform.claude.com/docs/en/api/messages/create).
