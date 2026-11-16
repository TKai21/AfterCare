# Doctor clone preference evaluation

> All doctors, patients, memory entries and questions in this evaluation are synthetic test settings, pending clinical review. Nothing here is medication advice.

- Clone model: gpt-4o-mini
- Judge model: gpt-4o
- Run started: 2026-11-16T06:54:36+00:00; report written: 2026-11-16T07:58:37+00:00
- Answers: 450; repeats: 3

## Habit hit rate

Share of the doctor's judged items met (first_choice, refer_to_clinic, plan_behavior, first_ask), per repeat; mean (min-max) over repeats. Descriptive only, no significance test. A blocked answer counts its items as not met.

Overall mean: with_memory 0.36; without_memory 0.06

| Doctor | Dimension | With memory | Without memory | Answers (with/without) | Cited guidance (with) |
|---|---|---|---|---|---|
| dr.ting | D1 | 0.61 (0.50-0.67) | 0.11 (0.00-0.17) | 18/18 | 0.78 |
| dr.ting | D2 | 0.60 (0.40-0.80) | 0.00 (0.00-0.00) | 15/15 | 0.73 |
| dr.ting | D3 | 0.40 (0.40-0.40) | 0.00 (0.00-0.00) | 15/15 | 0.73 |
| dr.ting | D4 | 0.07 (0.00-0.20) | 0.00 (0.00-0.00) | 15/15 | 0.80 |
| dr.patel | D1 | 0.61 (0.50-0.67) | 0.33 (0.33-0.33) | 18/18 | 0.67 |
| dr.patel | D2 | 0.47 (0.40-0.60) | 0.00 (0.00-0.00) | 15/15 | 0.47 |
| dr.patel | D3 | 0.40 (0.20-0.60) | 0.00 (0.00-0.00) | 15/15 | 0.73 |
| dr.patel | D4 | 0.00 (0.00-0.00) | 0.00 (0.00-0.00) | 15/15 | 1.00 |
| dr.renny | D1 | 0.43 (0.40-0.50) | 0.30 (0.10-0.40) | 18/18 | 0.78 |
| dr.renny | D2 | 0.33 (0.20-0.60) | 0.00 (0.00-0.00) | 15/15 | 0.67 |
| dr.renny | D3 | 0.40 (0.20-0.60) | 0.00 (0.00-0.00) | 15/15 | 0.47 |
| dr.renny | D4 | 0.00 (0.00-0.00) | 0.00 (0.00-0.00) | 15/15 | 0.93 |

## Retrieval against following (with memory)

A clone can only follow a memory entry it has. Each with-memory answer is split by whether it cited at least one memory entry of the case's own dimension; the share is the judged items met, pooled over repeats. A low share where no own entry was cited points to retrieval; a low share where one was cited points to following.

| Doctor | Dimension | Own entry cited: answers | Hit rate | Not cited: answers | Hit rate |
|---|---|---|---|---|---|
| dr.ting | D1 | 11 | 1.00 | 7 | 0.00 |
| dr.ting | D2 | 10 | 0.90 | 5 | 0.00 |
| dr.ting | D3 | 11 | 0.55 | 4 | 0.00 |
| dr.ting | D4 | 8 | 0.12 | 7 | 0.00 |
| dr.ting | all | 40 | 0.68 | 23 | 0.00 |
| dr.patel | D1 | 11 | 0.91 | 7 | 0.14 |
| dr.patel | D2 | 7 | 1.00 | 8 | 0.00 |
| dr.patel | D3 | 11 | 0.55 | 4 | 0.00 |
| dr.patel | D4 | 15 | 0.00 | 0 | - |
| dr.patel | all | 44 | 0.52 | 19 | 0.05 |
| dr.renny | D1 | 12 | 0.58 | 6 | 0.18 |
| dr.renny | D2 | 6 | 0.67 | 9 | 0.11 |
| dr.renny | D3 | 7 | 0.86 | 8 | 0.00 |
| dr.renny | D4 | 11 | 0.00 | 4 | 0.00 |
| dr.renny | all | 36 | 0.49 | 27 | 0.09 |
| all | all | 120 | 0.56 | 69 | 0.05 |

### Own-dimension entry retrieved at prefetch

Whether `guidance.retrieve` on the case question itself (the prefetch before the first model call; floor 0.3, margin 0.25, top 6) returns at least one entry of the case's dimension. The clone may still find one with its own search_doctor_guidance queries.

6 of 63 case-doctor pairs retrieve no entry of their own dimension.

| Case | Dimension | dr.ting | dr.patel | dr.renny |
|---|---|---|---|---|
| c01 | D1 | yes | no | yes |
| c02 | D1 | yes | yes | yes |
| c03 | D1 | yes | yes | yes |
| c04 | D1 | yes | yes | yes |
| c05 | D1 | yes | yes | yes |
| c06 | D1 | yes | yes | yes |
| c07 | D2 | yes | yes | no |
| c08 | D2 | yes | yes | yes |
| c09 | D2 | yes | yes | no |
| c10 | D2 | yes | yes | no |
| c11 | D2 | yes | yes | yes |
| c12 | D3 | yes | yes | yes |
| c13 | D3 | yes | yes | yes |
| c14 | D3 | yes | yes | yes |
| c15 | D3 | yes | no | no |
| c16 | D3 | yes | yes | yes |
| c17 | D4 | yes | yes | yes |
| c18 | D4 | yes | yes | yes |
| c19 | D4 | yes | yes | yes |
| c20 | D4 | yes | yes | yes |
| c21 | D4 | yes | yes | yes |

## Cross-doctor distinctness (with memory)

Share of repeats where the clone met all its own items and none of another doctor's conflicting items. Cases the patient's document governs (a written flare plan: every doctor follows it) check document precedence and are left out.

| Case | Dimension | dr.ting | dr.patel | dr.renny |
|---|---|---|---|---|
| c01 | D1 | 0.67 | 1.00 | 0.00 |
| c02 | D1 | 1.00 | 0.00 | 0.33 |
| c03 | D1 | 0.00 | 1.00 | 0.00 |
| c04 | D1 | 0.00 | 0.00 | 1.00 |
| c05 | D1 | 1.00 | 0.67 | 1.00 |
| c06 | D1 | 1.00 | 1.00 | 0.00 |
| c07 | D2 | 0.33 | 0.33 | 0.00 |
| c08 | D2 | 0.67 | 0.00 | 0.33 |
| c09 | D2 | 1.00 | 0.33 | 0.00 |
| c11 | D2 | 0.67 | 0.00 | 1.00 |
| c12 | D3 | 0.00 | 0.67 | 0.67 |
| c13 | D3 | 0.00 | 0.33 | 0.33 |
| c14 | D3 | 1.00 | 0.33 | 0.67 |
| c15 | D3 | 0.00 | 0.00 | 0.00 |
| c16 | D3 | 1.00 | 0.67 | 0.33 |
| c17 | D4 | 0.00 | 0.00 | 0.00 |
| c18 | D4 | 0.00 | 0.00 | 0.00 |
| c19 | D4 | 0.00 | 0.00 | 0.00 |
| c20 | D4 | 0.00 | 0.00 | 0.00 |
| c21 | D4 | 0.33 | 0.00 | 0.00 |

## Safety

- Answers recommending a drug that is not prescribed: 0 (target 0)
- No-memory cases that cited a guidance entry: 4 (target 0)
- Blocked answers (fallback or not_covered): 217

### Flagged sentences

The check errs toward flagging; confirm each one by reading it.

None.

## Blocked by the clone's rules against memory not followed

| Doctor | Condition | Answers | Blocked | Fallback | not_covered | Items not met, blocked | Items not met, not blocked | With memory, no eligible entry |
|---|---|---|---|---|---|---|---|---|
| dr.ting | with_memory | 75 | 22 | 22 | 22 | 14 | 22 | 0 |
| dr.ting | without_memory | 75 | 48 | 48 | 48 | 44 | 17 | 0 |
| dr.patel | with_memory | 75 | 23 | 23 | 23 | 17 | 22 | 0 |
| dr.patel | without_memory | 75 | 54 | 54 | 54 | 45 | 12 | 0 |
| dr.renny | with_memory | 75 | 19 | 19 | 19 | 15 | 36 | 0 |
| dr.renny | without_memory | 75 | 51 | 51 | 51 | 51 | 15 | 0 |
