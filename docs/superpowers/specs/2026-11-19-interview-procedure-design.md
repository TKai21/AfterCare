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

## 9. As built (2026-11-19)

- Switch: `INTERVIEW_PROCEDURE` (default on). The evaluation scripts turn it
  off, because their simulated patients answer the habit-only interview; the
  PortalPal re-evaluation with the procedure is still to do.
- Confirmation: the summary card has "That's right" and "Something is wrong".
  The second focuses the text box, and anything typed there is the correction
  (no separate box); the composer stays in place.
- Cost (4.6): the per-round judge already read only the open safety items and
  the terms a reply names, so nothing changed there. One runny-nose interview
  on a copy of the demo database: 9 questions, a summary and a close in 27
  model calls, about $0.008 at gpt-4o-mini prices (estimated from the size of
  the calls, not billed figures).
- Found on the way and fixed: a close habit that is only the tail of a
  sentence can no longer close a case; a clean denial may name what the
  clone's own question asked about ("No wheezing" after "... such as
  wheezing, cough, or shortness of breath?"); a citation marker no longer
  starts a line by itself.

## 10. Amended after the trial (2026-11-20)

Ten simulated patients (a model playing each, with its own facts and way of
talking) went through the procedure of section 9 on copies of the demo
database. No case was closed by the clone, four patients with no danger sign
got the emergency answer, and one was asked 16 questions. The team then chose
a shorter procedure whose fixed questions are the facts every case is
compared on. Changes:

- **Stages.** Safety, the problem, the doctor's follow-ups (at most 3), and
  medicines only when the build call says a medicine matters to the complaint.
  No closing "anything else?" question: it made patients retell everything;
  the summary's second button is "Add or correct something".
- **A reply answers its question.** In the procedure the reply to a question
  is stored as that item's answer, word for word, when the judge reads it as a
  reply. Replies are no longer read into other items, and "No." is no longer
  asked again. A short typed "I'm not sure" marks the item unsure at once.
- **Danger signs.** Two wordings of one sign are asked once, and a sign that
  only says "getting worse" is left to the question about that.
- **Denials** (amends 4.4 of the 2026-11-17 design). A list of known terms
  that ends in "or"/"nor" is denied as a whole ("no wheezing, cough, or
  shortness of breath"); every element must be a known term, and the things
  the clone's own question named count as known. Without the "or", with a
  hedge, or after anything that is not a denial, the emergency answer stays.

The same ten patients afterwards: two emergency answers without a danger sign
(was four), three cases settled by the clone (was none), 4.1 questions on
average in the interviews (was 6.8), 16 questions at most became 7. Still
open: a talkative patient who denies a danger sign among other remarks, and a
denial that names a symptom no list knows ("no trouble swallowing"), both get
the emergency answer; medicines were not asked in any of the ten, so a patient
who had stopped a medicine was not found out; a case closed although the
patient was unsure how bad the problem was; a symptom the document only
touches on was answered from the document without an interview.

### Second amendment (2026-11-20)

- Whether the patient takes their medicines as directed is asked whenever the
  document lists a medication (in the ten cases the build call never judged
  medicines relevant, so a patient who had stopped one was not found out);
  what else they took stays conditional.
- A case is not closed when the patient could not say how long, how bad, or
  whether it is changing.
- At the end of an interview the clone closes in the doctor's own words or
  hands the case over. It no longer delivers an answer it composed from what
  the patient said: one run told a patient "Your headache may be related to
  not taking colchicine".

Third run of the ten patients: three emergency answers without a danger sign,
one case closed by the clone, five handed to the doctor with a confirmed
summary, 4.3 questions on average. The emergency answers differ from run to
run because the follow-ups asked differ; the remaining cause is the same in
all three: a natural denial the rules do not accept ("No, I have not had any
additional symptoms such as wheezing, cough, or shortness of breath").

### Third amendment (2026-11-21): ask, do not guess

Three runs left two to four of ten patients with the emergency answer for a
danger sign they had denied in ordinary words. Each wording the denial rules
were widened for was followed by another, and every widening raised the risk
of reading a real emergency as a denial. The team chose to stop reading
denials and ask instead.

- **The danger check.** In the procedure, when a reply names a danger sign and
  is not a plain denial, the program may show one question in place of the
  emergency answer: "Are you having this right now?", the signs, "If you are,
  call 911 or follow the emergency instructions in your discharge document",
  and two buttons, "Yes, I am" and "No, I am not". No model call is made. The
  reply is held, the composer is disabled until a button is pressed, and the
  doctor's card shows the held reply and the unanswered check if it times out.
- **When it is asked** (`safety.check_possible`): only when every sentence
  that names a danger sign also has a word of denial. A sign with no such word
  in its sentence ("Yes, I have chest pain", "Not sure. Maybe some chest
  pain."), a self-harm or suicide term, a term that is itself a negation
  ("can't breathe"), and text with look-alike letters get the emergency answer
  at once, as before. So does a first message, and a reply outside the
  procedure.
- **Yes**, or anything typed in place of a button: the fixed emergency answer
  with no model call, and the case goes to the doctor with the held reply, the
  check and the patient's answer.
- **No**: the interview goes on with the held reply. The signs are stored on
  the round as denied by the patient's own answer; this, not the reply judge's
  reading, is their status. The summary the patient confirms and the doctor's
  card name them. At the close gate the button stands in for the plain denial
  of that sign in that reply only; the model must still read it as absent.
- **Plain denials** ("No fever.", "No chest pain or fainting") still pass
  without a check, by the rules of 2026-11-17. The list rule of the first
  amendment ("no A, B, or C") is withdrawn: such a reply now gets the check.

Fourth run of the ten patients: no emergency answer without a danger sign
(was three), two checks shown, both answered No; the patient who has
shortness of breath got the emergency answer at once, with no check. Two cases
closed by the clone, five handed to the doctor with a confirmed summary, 5.4
questions on average. Still open: a "call the clinic" sign denied in ordinary
words ("I haven't had any headache, fever, vomiting, or diarrhea") blocks the
close with the reason "Warning sign: you have diarrhea", which misleads the
doctor; and a symptom the document only touches on is still answered from the
document without an interview.

Calls made by Claude: the rule for when the check may be asked; typed words
during a check get the emergency answer; the check is limited to the
procedure; the button's reach at the close gate.

### Fourth amendment (2026-11-21): medicine questions follow the complaint

A team member with a runny nose was asked "Have you been taking colchicine and
naproxen as your discharge instructions say?". The question pulls the
conversation back to the gout document and reads as the clone not listening.
The second amendment had made it unconditional because the build call never
judged medicines relevant; the cause was that the call was never shown the
document.

- **The decision.** The build call now gets the document's text (first 4000
  characters), its problems and instructions, and its medicine names, and
  returns `document_related`: the complaint is a symptom, flare or
  complication of what the document treats, or a known side effect of one of
  its medicines. The program also decides "related" by itself when the message
  names a document medicine or problem (`interview.names_document`). Anything
  but a clear yes is "not related".
- **Related** (toe pain after a gout flare): as before, whether the medicines
  are taken as the discharge instructions say, and what else was taken.
- **Not related** (a runny nose): one question in fixed words, "What medicines
  are you taking right now, including anything you took for this?", with no
  word about the document. A habit of the doctor that asks about medicines is
  not asked beside it.
- **The doctor's side.** The interview keeps the document's medicine names,
  and the summary the doctor reads adds "The discharge document lists: ..."
  under the patient's answer, so a medicine the patient left out is seen. The
  doctor judges it; the clone does not ask again.

Checked with the real model on the demo document, twice each: toe pain, a gout
flare, a swollen red foot, stomach pain since the pain pills and diarrhea were
read as related; a runny nose, a cough, a headache, burning urination and
constipation as not related. In the app the runny nose, the cough and the
headache got the new question. Still open: the related complaints were
answered from the document without an interview in these runs, so the related
questions were not seen end to end with the real model.

Calls made by Claude: the wording of the one question; "not related" as the
fallback; the doctor reads the list instead of the clone comparing it.

### Fifth amendment (2026-11-21): a new symptom is not an answer

Since the first amendment a reply is filed as the answer to its question, word
for word, when the judge reads it as a reply. The judge calls anything about
the patient's health a reply, so "My toe still hurts a lot after four days."
typed at "What medicines are you taking right now?" became the answer to the
medicine question and the interview went on to the summary.

- The reply judge returns one more field, `answers_question`: whether the
  newest message gives what the last round's questions ask for.
- In the procedure a reply the judge reads as not answering its question and
  as raising something else is handled as a side question (4.3): the agent
  answers it from the document or hands it to the doctor as a question of its
  own, nothing is filed, and the open question is asked again. The cap of
  three side questions applies.
- Both readings are needed. A short answer the judge misreads as "does not
  answer", an answer with something added ("Only colchicine. Also my toe hurts
  again."), and a judge that does not return the field all stay answers, as
  before.

Checked with the real model: twelve replies at the medicine question in the
app and sixteen question and reply pairs from the other steps on the judge
alone were all read as intended, with no real answer turned away. Before the
change two of the twelve replies were filed as the medicine answer.

