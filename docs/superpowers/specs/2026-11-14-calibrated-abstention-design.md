# AfterCare - Calibrated Abstention Design

Date: 2026-11-14
Status: approved by Kai (go-ahead without a question round)
Builds on: `2026-11-14-guidance-rules-design.md`, `2026-11-14-clone-drafts-design.md`. Part 3 of 3 of the clone upgrade.

## 1. Intent

The clone decides on its own whether to answer or send a question to the doctor, and the server's checks (citations exist, numbers are grounded) catch only some wrong answers: an answer can cite real evidence and still be wrong or incomplete. The guidance similarity cut (0.45) was picked by hand. Nothing says how often a patient gets a wrong answer.

This change adds a **confidence gate with a statistical guarantee**. Every answer the clone would give is scored by a separate verifier call. An answer below a threshold goes to the doctor instead. The threshold is chosen by **conformal risk control** (Angelopoulos et al., 2022) on a labeled calibration set, so that the expected rate of wrong answers delivered to patients is at most a chosen alpha, for new questions drawn like the calibration questions, whatever the model or the score.

Success: an evaluation report that states, for the chosen alpha, the threshold, the measured wrong-answer rate and answer rate on held-out test questions, and the risk-coverage curve; the server applies the threshold.

## 2. Goals

1. A verifier that scores an answer's support from its cited evidence, from 0 to 1.
2. A labeled clone evaluation set: patient questions with the expected behavior (answer with key facts, or escalate), split into calibration and test.
3. A script that runs the real clone end to end on the set, grades answers, computes the conformal threshold on the calibration split, and reports risk and coverage on the test split. Results go to `eval/clone_calibration.json`.
4. The server reads that file (or `CLONE_MIN_CONFIDENCE`) and gates answers; without either, behavior is unchanged.

## 3. Non-goals and anti-goals

Non-goals: calibrating per doctor or per topic (too little data); showing confidence to patients; changing the emergency path.

Anti-goals:

- **A guarantee claimed beyond what it covers.** The bound holds in expectation over exchangeable questions, with a set of this size. The report says so, and says the set is written by the team, not drawn from real patients.
- **The gate makes an emergency wait.** It never runs on emergency answers, and a verifier failure never blocks the emergency answer.
- **A broken verifier lets answers through.** With the gate on, a verifier failure is treated as low confidence (fail closed).

## 4. Approaches

- **A. Verifier score with conformal risk control (chosen).** Any score works for validity; a verifier that reads the answer against its cited evidence is a good score because it sees what the number checks cannot. One extra model call per answered question.
- **B. Calibrate the retrieval similarity cut.** Cheap, but only covers answers from the doctor's advice, and similarity says nothing about whether the composed answer is right.
- **C. The model's self-reported confidence inside the answer call.** No extra call, but the model grading itself in the same pass is the weakest score, and it changes the agent's output contract.

## 5. Verifier

`confidence.score_answer(llm, question, answer, sources)`: one JSON-mode call with the patient question, the answer with its markers, and each cited source's text. The model rates whether every statement is supported by its cited source and the answer addresses the question, returning `{"score": 0-100, "unsupported": [...]}`. The score divided by 100 is the confidence. An unreadable reply raises ValueError.

## 6. Conformal risk control

Calibration set of n questions. For question i: `answered_i` (the clone gave kind `answer`), `s_i` its confidence, `wrong_i` (graded wrong). The loss at threshold t is `L_i(t) = 1 if answered_i and s_i >= t and wrong_i else 0`, which is non-increasing in t and bounded by 1. With `R(t)` the mean loss, the threshold is

    t* = min { t : n/(n+1) * R(t) + 1/(n+1) <= alpha }

over the candidate thresholds (the observed scores and 1.01, which answers nothing). Then E[loss on a new question] <= alpha. If even answering nothing fails (n < 1/alpha - 1), the script refuses.

Coverage, the cost: the share of questions that expect an answer that are answered correctly and pass the gate.

## 7. Evaluation set

`eval/clone_questions.json`, about 100 questions for the two demo patients (kidney stones with Dr. Renny's clone, gout with Dr. Ting's and Dr. Patel's), each `{id, patient, doctor, question, expect: "answer" | "escalate", facts: [...], split: "calibration" | "test"}`. `facts` are the key facts a correct answer must state. Escalate questions are things no approved source answers (and near misses that a plausible but unsupported answer would get wrong). `eval/clone_guidance.json` adds the doctors' fictional earlier advice that some questions rely on, including questions that need two entries combined.

Grading: escalate questions are wrong if answered. Answer questions are graded by a judge call that checks the answer states every fact and contradicts none; the judge's verdicts are saved in the results for review.

## 8. Script and server

`scripts/calibrate_clone.py` (dev only, real model from `.env`, never in pytest): builds a temporary SQLite database with the demo seed and the eval guidance (rules read by the real model), runs each question through `/api/ask` in-process (the gate off), scores answered questions with the verifier, grades, computes `t*` on calibration for `--alpha` (default 0.1), reports test risk and coverage, writes `eval/clone_calibration.json` (`threshold`, `alpha`, `n`, model names, date, per-question results) and a risk-coverage figure.

Server: `Settings.clone_min_confidence` from `CLONE_MIN_CONFIDENCE`, else the calibration file's threshold when its chat model matches `LLM_MODEL`, else none. With a threshold, `/api/ask` scores each `answer` result; below it, or on a verifier failure, the patient gets the fixed not-covered sentence and the question goes to the doctor's queue with reason "Low confidence".

## 9. Testing

Unit: `crc_threshold` (worked examples, the infeasible case, monotonicity), the verifier parse, calibration file loading (model mismatch ignored). Routes: gate off by default, a low score and a verifier failure escalate, a high score passes, emergencies are never scored. Script: the pure parts (grading of escalate questions, report math) with the fake model.
