# Can a recovery model survive a change of router?

This experiment imports generated CSV files through Python's standard library. A router chooses comma or semicolon. The importer first decodes with UTF-8. After a parser error it can switch **one** setting and retry the original bytes once. Returning the exact source records, including all Unicode characters and embedded newlines, earns one success. An unresolved file remains in the denominator and goes to review.

## Questions fixed before scoring

1. **Does the recovery training population matter after replacing a router?** With a deployed tree router and a tree recovery model that sees only “failed”, compare training on that tree's failures against training on the original hint router's failures. Both receive the same number of labeled files. Report paired improvements, losses and ties, across all files and by family. Seed 17 is the primary recipe. Repeat with seeds 29 and 43 for sensitivity. Use the same 256 files for each seed.
2. **Can a better observation substitute for recollecting failures?** Repeat with the actual parser error class available to the recovery model. Publish the ceiling for any fixed recovery action when all failures look alike, the ceiling when error classes are visible, and the ceiling with a separate best action for each file.
3. **Which stages benefit from learning?** Cross the training router, deployed router, recovery architecture and observable signal. Compare against following the supplied hint, skipping recovery, a coded error handler and a fully coded header-plus-error handler. Publish each seed. Treat comparisons outside the primary question as diagnostic.

## Data, budgets and controls

The generator assigns factors independently by a fixed pseudorandom draw within each family. Half the files use a semicolon, one quarter have a stale delimiter hint, about one fifth use Windows-1252, and one sixteenth have a corrupted numeric value. Some files need two changes or have no possible repair under the available actions. The header is deliberately informative. The quoted family adds escaped quotes, delimiters inside values and embedded newlines. It changes the body the parser encounters, while preserving the router's header signal.

| Partition | Files | Use |
| --- | ---: | --- |
| Fit | 128 plain | Train the routing tree and Google BERT-Tiny from identical delimiter labels |
| Collect | 256 plain | Run each fitted router and the supplied hint. For detected failures only, try both legal repairs and retain cases with exactly one content-correct action |
| Development | 96 plain | Report behavior on unseen files with the familiar layout |
| Test | 128 plain + 128 quoted | Locked evaluation, including a layout held out of fitting and collection |

IDs and payload bytes must be disjoint between partitions. Source rows are held outside the policy inputs. Oracle labels can be used during collection and grading. A deployed repair specialist only sees the observed parser status or the single generic failure token. A router sees the declared hint and two counts from header bytes. A router referral falls back to the supplied hint. A recovery referral spends no retry. The coded router selects the most frequent delimiter in the header and falls back to the hint on a tie. The coded recovery uses the encoding switch after a decode error and the delimiter switch after a shape error. A value error goes to review.

For each seed, take the same number of uniquely repairable collection failures per router using the smallest eligible pool. Traverse files in SHA-256 order of ID before taking this prefix. Keep the resulting frequencies and selected IDs. Each tree/BERT pair uses identical task records. Trees use the workshop defaults. BERT updates the pinned Google encoder and its decision head with four epochs, batch size 16 and learning rate 0.0003. Freeze all trained artifacts before grading either evaluation split. Reuse a checkpoint when its training data, recipe, task and seed are identical. Collection behavior helped design this fixture. Neither evaluation split selects a checkpoint.

There are 48 policies per seed. Each of the hint, tree and BERT routers runs with 12 trained recovery policies (three training sources × two observations × two architectures) and three controls (no retry, coded handler, best obtainable result). The fully coded header router runs the three controls. Apply every policy to every file in the same split. Count first-pass and final exact outputs, errors reaching stage two, retries, recoveries, wrong accepted outputs, and opportunities under the one-retry budget. Record actual actions and status for each file. The hindsight control may read the expected records. Its results describe the action budget.

For the primary contrast calculate a paired percentile interval across 2,000 resamples of whole test files within each family, using random seed 814. Report seed-specific counts and avoid counting the same files three times as additional evidence. This is a controlled study of a specific parser and synthetic fault distribution. Changing the catalogs, importer or error mix can change the result.
