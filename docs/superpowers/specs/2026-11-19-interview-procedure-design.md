# Interview procedure - design

Date: 2026-11-19. Approved by the team in conversation. Amends
`2026-11-17-clone-interview-design.md`; everything there not named here stays.

## 1. Intent

The team wants the doctor's clone to have a complete procedure for asking: the
same steps every time, in the same order, followed to the end. Today the
questions are whatever the doctor's similar past cases happen to contain. A
real run on 2026-11-19 (runny nose, Dr. Okafor) asked four questions, never
asked how bad it was or about medication, did not wrap up, did not show the
patient what it understood, and treated the patient's next two messages as new
questions.

## 2. Goals, non-goals, anti-goals

Goals
- Five fixed stages, always in this order, always completed: safety, the
  problem, the doctor's follow-ups, medication, wrap-up.
- Inside a stage the doctor's own wording is used when the doctor has it.
- A side question does not end the interview: it is answered, then the
  interview goes on.
- The patient sees and confirms what the clone understood before the case is
  closed or handed to the doctor.
- The patient always knows where they are and when it is over.

Non-goals
- Doctors editing the stages or the fixed questions (later).
- Medication advice from the doctor's medication habits (the next design; this
  one only collects what that needs).

Anti-goals
- A fixed question that asks what the patient already said.
- The procedure delaying an emergency: a danger sign still interrupts at once.
- A longer interview that costs much more per conversation than today.

## 3. Approaches considered

- **Chosen: fixed stages, doctor's questions inside them.** Predictable and
  complete, still in the doctor's voice.
- Stages that are skipped when they look irrelevant: closer to a real doctor,
  but the length varies and the team asked for a complete procedure.
- A full fixed questionnaire: complete, but the "fixed questions" the team
  rejected on 2026-11-17.

## 4. Design

### 4.1 Stages and items

Every item has a `stage` and the interview asks stage by stage. Core items
exist in every interview; habit items come from the doctor's similar cases as
today.

| Stage | Items | Source |
|---|---|---|
| 1 safety | danger signs relevant to the complaint (at most `INTERVIEW_MAX_SAFETY`) | as today (4.7 item 7 of the 2026-11-17 design) |
| 2 problem | core: `how long`, `how bad`, `getting better or worse` | fixed, see 4.2 |
| 3 follow-ups | the doctor's ask habits not used in another stage (at most `INTERVIEW_MAX_ITEMS`) | as today |
| 4 medication | core: `taking prescribed medicines as directed`, `anything else taken for this` | fixed, see 4.2 |
| 5 wrap-up | core: `anything else for the doctor` | fixed |

Stage 4's first item is left out when the document lists no medication.

### 4.2 Core items and wording

The build call (one call, as today) also maps the doctor's ask habits to core
items: its JSON gains `"core": {"how long": [rule ids], ...}`. A core item
with a mapped habit is asked in the doctor's words (verbatim, as today);
otherwise with its fixed question:

- how long: "How long has this been going on?"
- how bad: "How bad is it right now: mild, moderate or severe?"
- getting better or worse: "Is it getting better, getting worse, or staying
  the same?"
- taking prescribed medicines as directed: "Have you been taking {names} as
  your discharge instructions say?" ({names}: the document's medication names,
  joined with "and"; more than three becomes "your medicines")
- anything else taken for this: "Have you taken anything else for this, such
  as an over-the-counter medicine?"
- anything else for the doctor: "Is there anything else you want
  {doctor} to know?"

As today, the extractor fills an item the patient already answered, so it is
not asked. The wrap-up item is never filled from earlier messages.

A safety term that does not read as a thing a patient can have (it has a verb,
for example "symptoms become acutely worse") is asked as "Is this happening
now: {term}?" instead of "Have you had any {term}?".

### 4.3 Completing the procedure

- Rounds: the interview ends when no item is unknown. `INTERVIEW_MAX_ROUNDS`
  becomes a safety cap, default 16.
- A side question (the reply judge says it is not a reply, as today) is
  answered by the agent in the same turn, then the next item is asked under
  the answer. The interview stays open. At most 3 side questions; a fourth
  ends the interview as today.
- "Not sure" works as today.
- A wrap-up reply that adds something becomes `other_concerns`, so the case
  goes to the doctor with it (the close gate already does this).
- A danger sign that is present or unclear ends the interview at once, as
  today.

### 4.4 Confirmation

When no item is unknown, the patient sees "Here is what I understood" with the
item list by stage, and two buttons: "That's right" and "Something is wrong".
- "That's right": the agent runs as today (answer, close, or hand off).
- "Something is wrong": a text box; the correction is stored as a round and
  the case goes to the doctor with the correction on top.
Until the patient confirms, the composer is replaced by these two buttons. The
doctor's inbox shows whether the patient confirmed.

### 4.5 What the patient sees

- Progress: "Step 2 of 5 - Your symptoms" above the question (stage names:
  Safety check, Your symptoms, A few more questions, Your medicines, Anything
  else).
- "So far" groups items by stage.
- After the end, one line under the last message: "That's everything I needed.
  You can ask another question any time." A later message is a new question,
  as today.

### 4.6 Cost

A longer interview must not cost much more. Per round the judge reads only the
asked item's terms and the terms the reply names (today: every warning term in
batches of 10); the full batched reading runs once, at the end, for the close
gate. Measured before and after with `scripts/run_with_spend.py` on the
runny-nose interview; the target is at most 1.5 times today's cost per
interview.

## 5. Verification

- Tests: stage order; core items present; a mapped habit replaces a fixed
  question; a filled item is skipped; a side question keeps the interview
  open; the confirmation step both ways; the cap.
- A real run on a copy of the demo database: runny nose, toe pain and a cough,
  transcripts attached to the pull request, with the cost of each.
- Screens at 1100 and 390 px: a question in each stage, the confirmation, and
  the end line.

## 6. Left to the builder

The build prompt wording for the core mapping, the rule that tells whether a
safety term has a verb, and the CSS of the confirmation card (it must match
the existing clarify card).

## 7. Decided by the team

A complete, fixed procedure that the clone follows to the end (2026-11-19).

## 8. Calls made by Claude

The five stages and the six core items; fixed wording as the fallback; side
questions answered in place, at most three; the confirmation step; the cap of
16; the judge saving in 4.6.
