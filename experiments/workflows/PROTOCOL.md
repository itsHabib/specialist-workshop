# Workflow experiment protocol — fixed before model evaluation

Two tasks, one 1.5B pinned base, two fresh supervised QLoRA adapters. Keep the
original CI experiment and its failed comparisons unchanged.

1. Blox arithmetic screening: classify explicit prescription arithmetic as
   consistent, time-overrun, distance-conflict, or needs-context. Training uses
   repeated work, single benchmarks and shuttle distances; development uses
   pace-derived work; final cases use compound blocks and combined distances.
   Scenario families stay in one split. Numbers and authored language are
   synthetic; shared arithmetic primitives are intentional. Existing Blox
   counterexamples are exposed diagnostic evidence, never a fresh holdout.
   A deterministic parser/calculator is the useful baseline. Agreement checks
   are arithmetic only, not coaching clearance or a plan-wide safety verdict.
2. CI evidence practice: inspect a failed log, optionally inspect same-commit
   history, then write an advisory disposition. Each episode gets a disposable
   directory with immutable repository bytes and private evidence fixtures.
   Tool transitions are authored, finite and checked; no arbitrary commands,
   network, actual CI retry, repository repair or Gate actions. Signatures and
   wrapper failures come from Workbench's current ci-classify implementation.
   Distinct signature families are assigned to train/development/final sets.
   Family variants remain together. A rule policy and training-majority policy
   execute through exactly the same environment as model policies.

Fixed recipe: existing worker, 200 iterations, batch 2, 8 layers, learning rate
2e-5, seed 17, assistant-only loss, 1024 token maximum. No tuning after final
results. Save every raw response, input, transition, data/prompt/checkpoint hash,
wall time and available memory telemetry. Train only train, monitor only dev.
No teacher labels, RL, GRPO, or frontier-distillation claim.

Score classification and strict accepted correctness; score environment final
state, forbidden/invalid actions, premature or incorrect dispositions,
escalation, unnecessary escalation, failed episodes needing human correction,
and trajectory length. Report generation and whole-run latency separately.
Local energy/hardware monetary cost is unmeasured. A logged-in Claude CLI may
supply a tool-free frontier reference on a fixed subset: first two final cases
per arithmetic label and first final CI episode per expected disposition.
This smaller reference is descriptive; compare local policies on that same
subset. Preserve CLI model identity and usage estimates, not invoice claims.

Do not use final inputs in further correction/training in this experiment.
Final authored sets are unseen by gradient updates, not independent external
qualification. The author knows the generator, rules and grader. Kill condition:
no model integration unless it adds useful correctness beyond the deterministic
baseline without more harmful decisions. A failed specialization result is valid.

Source anchors (read-only): Workbench cmd/gate/internal/verify/ciclassify.go and
ciclassify_test.go; RoxIQ blox/internal/blox/genplan/validate.go; workout-lab
2026-09-07/experiment/{comparison-report.md,failure-cases.json}. Ivy is deferred:
no independently reviewed learner/hint labels were established in this task.
DeepSeek v4 MEAP chapter 7 supports verified demonstrations before reward training.
