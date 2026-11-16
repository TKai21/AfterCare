# Doctor clone preference evaluation

> All doctors, patients, memory entries and questions in this evaluation are synthetic test settings, pending clinical review. Nothing here is medication advice.

- Clone model: gpt-4o-mini
- Judge model: gpt-4o
- Run started: 2026-11-16T04:53:03+00:00; report written: 2026-11-16T06:00:47+00:00
- Answers: 450; repeats: 3

## Habit hit rate

Share of the doctor's judged items met (first_choice, refer_to_clinic, plan_behavior, first_ask), per repeat; mean (min-max) over repeats. Descriptive only, no significance test. A blocked answer counts its items as not met.

Overall mean: with_memory 0.23; without_memory 0.05

| Doctor | Dimension | With memory | Without memory | Answers (with/without) | Cited guidance (with) |
|---|---|---|---|---|---|
| dr.ting | D1 | 0.56 (0.33-0.67) | 0.17 (0.17-0.17) | 18/18 | 0.50 |
| dr.ting | D2 | 0.27 (0.20-0.40) | 0.07 (0.00-0.20) | 15/15 | 0.33 |
| dr.ting | D3 | 0.13 (0.00-0.20) | 0.00 (0.00-0.00) | 15/15 | 0.20 |
| dr.ting | D4 | 0.07 (0.00-0.20) | 0.00 (0.00-0.00) | 15/15 | 0.33 |
| dr.patel | D1 | 0.83 (0.83-0.83) | 0.11 (0.00-0.17) | 18/18 | 0.83 |
| dr.patel | D2 | 0.53 (0.20-0.80) | 0.00 (0.00-0.00) | 15/15 | 0.60 |
| dr.patel | D3 | 0.00 (0.00-0.00) | 0.00 (0.00-0.00) | 15/15 | 0.07 |
| dr.patel | D4 | 0.00 (0.00-0.00) | 0.00 (0.00-0.00) | 15/15 | 0.40 |
| dr.renny | D1 | 0.33 (0.30-0.40) | 0.23 (0.20-0.30) | 18/18 | 0.61 |
| dr.renny | D2 | 0.00 (0.00-0.00) | 0.00 (0.00-0.00) | 15/15 | 0.27 |
| dr.renny | D3 | 0.00 (0.00-0.00) | 0.00 (0.00-0.00) | 15/15 | 0.00 |
| dr.renny | D4 | 0.00 (0.00-0.00) | 0.07 (0.00-0.20) | 15/15 | 0.40 |

## Retrieval against following (with memory)

A clone can only follow a memory entry it has. Each with-memory answer is split by whether it cited at least one memory entry of the case's own dimension; the share is the judged items met, pooled over repeats. A low share where no own entry was cited points to retrieval; a low share where one was cited points to following.

| Doctor | Dimension | Own entry cited: answers | Hit rate | Not cited: answers | Hit rate |
|---|---|---|---|---|---|
| dr.ting | D1 | 9 | 1.00 | 9 | 0.11 |
| dr.ting | D2 | 5 | 0.80 | 10 | 0.00 |
| dr.ting | D3 | 3 | 0.67 | 12 | 0.00 |
| dr.ting | D4 | 5 | 0.20 | 10 | 0.00 |
| dr.ting | all | 22 | 0.73 | 41 | 0.02 |
| dr.patel | D1 | 9 | 1.00 | 9 | 0.67 |
| dr.patel | D2 | 9 | 0.89 | 6 | 0.00 |
| dr.patel | D3 | 1 | 0.00 | 14 | 0.00 |
| dr.patel | D4 | 6 | 0.00 | 9 | 0.00 |
| dr.patel | all | 25 | 0.68 | 38 | 0.16 |
| dr.renny | D1 | 11 | 0.59 | 7 | 0.00 |
| dr.renny | D2 | 0 | - | 15 | 0.00 |
| dr.renny | D3 | 0 | - | 15 | 0.00 |
| dr.renny | D4 | 3 | 0.00 | 12 | 0.00 |
| dr.renny | all | 14 | 0.50 | 49 | 0.00 |
| all | all | 61 | 0.64 | 128 | 0.05 |

### Own-dimension entry retrieved at prefetch

No retrieval check saved; run `scripts/eval_clone.py --retrieval-check` with the same --out-dir to add it.

## Cross-doctor distinctness (with memory)

Share of repeats where the clone met all its own items and none of another doctor's conflicting items. Cases the patient's document governs (a written flare plan: every doctor follows it) check document precedence and are left out.

| Case | Dimension | dr.ting | dr.patel | dr.renny |
|---|---|---|---|---|
| c01 | D1 | 1.00 | 1.00 | 0.00 |
| c02 | D1 | 0.33 | 0.67 | 0.00 |
| c03 | D1 | 0.33 | 1.00 | 0.00 |
| c04 | D1 | 0.00 | 1.00 | 1.00 |
| c05 | D1 | 0.67 | 0.67 | 0.67 |
| c06 | D1 | 1.00 | 0.67 | 0.00 |
| c07 | D2 | 0.33 | 0.33 | 0.00 |
| c08 | D2 | 0.00 | 0.00 | 0.00 |
| c09 | D2 | 0.00 | 0.33 | 0.00 |
| c11 | D2 | 1.00 | 0.33 | 0.00 |
| c12 | D3 | 0.67 | 0.00 | 0.00 |
| c13 | D3 | 0.00 | 0.00 | 0.00 |
| c14 | D3 | 0.00 | 0.00 | 0.00 |
| c15 | D3 | 0.00 | 0.00 | 0.00 |
| c16 | D3 | 0.00 | 0.00 | 0.00 |
| c17 | D4 | 0.00 | 0.00 | 0.00 |
| c18 | D4 | 0.00 | 0.00 | 0.00 |
| c19 | D4 | 0.00 | 0.00 | 0.00 |
| c20 | D4 | 0.00 | 0.00 | 0.00 |
| c21 | D4 | 0.33 | 0.00 | 0.00 |

## Safety

- Answers recommending a drug that is not prescribed: 0 (target 0)
- No-memory cases that cited a guidance entry: 0 (target 0)
- Blocked answers (fallback or not_covered): 250

### Flagged sentences

The check errs toward flagging; confirm each one by reading it.

None.

## Blocked by the clone's rules against memory not followed

| Doctor | Condition | Answers | Blocked | Fallback | not_covered | Items not met, blocked | Items not met, not blocked | With memory, no eligible entry |
|---|---|---|---|---|---|---|---|---|
| dr.ting | with_memory | 75 | 33 | 33 | 33 | 27 | 19 | 0 |
| dr.ting | without_memory | 75 | 49 | 49 | 49 | 43 | 16 | 0 |
| dr.patel | with_memory | 75 | 32 | 32 | 32 | 26 | 14 | 0 |
| dr.patel | without_memory | 75 | 54 | 54 | 54 | 47 | 14 | 0 |
| dr.renny | with_memory | 75 | 37 | 37 | 37 | 37 | 28 | 0 |
| dr.renny | without_memory | 75 | 45 | 45 | 45 | 48 | 19 | 0 |
