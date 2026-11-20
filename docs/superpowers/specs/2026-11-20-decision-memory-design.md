# Decision memory - design

Date: 2026-11-20. Status: first version built on 2026-11-21, reduced after the replay: see
section 9, which replaces 4.2, 4.4 and 4.6 where they differ.

## 1. Intent

The team's goal for the clone is to learn how the doctor judges, not only what
the doctor says. Today the clone keeps the doctor's words in pieces: questions
the doctor asks, how the doctor closes, signs the doctor wants to hear about,
and advice. Nothing links a patient's situation to what the doctor decided.
The clone knows Dr. Okafor says "If your symptoms persist, please call the
office" but not when she says it.

This design records each case as a pair, the situation and the doctor's
decision, and uses the pairs to propose a decision for the next similar case.
It also covers the doctor's medication habits: they are the decisions that
involve a drug.

## 2. Goals, non-goals, anti-goals

Goals
- Every case a doctor settles is stored as (situation, decision).
- For a new case the doctor sees the clone's proposed decision, the past cases
  it comes from, and a drafted reply, and can approve with one click.
- The doctor's corrections change the next proposal.
- When the doctor answers a case by asking the patient something, the clone
  asks that itself next time.
- A measured answer to "how often does the clone propose what the doctor
  decided?".

Non-goals
- Medical reasoning beyond "similar situation, same decision".
- Learning across doctors.
- Doctors editing the list of decision kinds (later).

Anti-goals
- The clone tells a patient to start, stop, change or swap a medicine without
  the doctor's approval of that message.
- A proposal with no past case behind it is presented as the doctor's habit.
- The clone acts alone on a kind of case the doctor has not approved several
  times.

## 3. Approaches considered

- **Chosen: case memory.** Store the pairs, find the nearest past cases, and
  propose their decision. It needs few cases to be useful, every proposal can
  show its evidence, and a wrong one is corrected by one more case.
- Decision rules written by a model from the cases ("more than 3 days and
  fever: come in"). Readable, but with a dozen cases the rules are guesses,
  and a wrong rule is silently applied to everyone.
- Fine-tuning a model on the doctor's replies. Needs far more data than one
  doctor has, and cannot show why.

Rules can be added later as a summary of the cases; the cases stay the source.

## 4. Design

### 4.1 The situation

The interview procedure (2026-11-19, amended 2026-11-20) already produces it:
the complaint, the danger signs denied or not, how long, how bad, better or
worse, the doctor's follow-ups, and medicines when asked, each with the
patient's words, and whether the patient confirmed the summary. A case whose
summary the patient did not confirm, or corrected, is stored with that flag
and is not used as evidence for proposals.

### 4.2 The decision

When the doctor sends a reply to a case, one small model call reads the reply
and names its kind, and the doctor sees it as a tag next to Send and can change
it before sending:

| Kind | Meaning |
|---|---|
| watch | Reassure; self-care with no drug; when to come back |
| follow document | Do what the discharge document already says |
| contact clinic | Call or message the clinic, not urgent |
| come in | Book a visit or be seen |
| urgent | Go to urgent care or the ER |
| medication | Start, stop, change, swap or skip a medicine |
| ask more | The doctor asks the patient a question |
| other | None of these |

A reply can have "medication" together with one other kind.

### 4.3 What is stored

A new table `case_decisions`: the doctor, the care-team question it settles,
the complaint, the situation (the interview items as stored), the kinds, the
doctor's reply, how it was made (the clone's draft unchanged, edited, or
written by the doctor), whether the patient confirmed the summary, and an
embedding of "complaint + the core facts".

### 4.4 The proposal

When a case reaches the doctor's inbox:

1. Find this doctor's nearest past cases (at most 5) above a similarity floor.
2. The proposed kind is the most common kind among them, weighted by
   similarity; none above the floor means no proposal ("No similar case yet").
3. The existing draft call (`drafts.py`) gets the past cases as examples:
   their situations, kinds and the doctor's replies.

The inbox card shows, above the draft: the situation by stage, "Proposed:
watch", and "Based on 3 of your past cases" which opens them. The buttons are
Approve and send, Edit, and Write my own.

### 4.5 Learning what to ask

A reply of kind "ask more" is the doctor asking for something the interview
did not collect. Its question is stored as an "ask" habit of that doctor (the
rule reader already does this) and linked to the case, so the interview's
follow-up stage picks it up for similar complaints. No new mechanism.

### 4.6 Acting alone

For a kind of case, the clone may send its reply without waiting only when
all hold:
- the kind is watch, follow document or contact clinic;
- the 3 nearest past cases all have that kind, each approved by the doctor
  with the draft unchanged or lightly edited;
- the patient confirmed the summary, every danger sign asked was denied, and
  the close gate passes as today;
- the doctor has this turned on (off by default, one switch on the doctor's
  page).

Such a reply is marked "sent by your clone" in the inbox, where the doctor can
follow up. Medication, come in, urgent and ask more always wait for the doctor.

### 4.7 Starting from history

A doctor's existing question and answer pairs (for Dr. Okafor, the imported
portal messages) become case records by one model call per pair that names
the kind; the situation is only what the patient's message says. They are
marked "history" and count as evidence for proposals but not toward 4.6.

## 5. Verification

- Replay: take the doctor's history, hide one case at a time, and ask for a
  proposal from the rest. Report how often the proposed kind matches the
  doctor's, per kind, and how often there is no proposal. This is the number
  to show for "the clone learns the doctor's judgement".
- Tests for the kind reader, the nearest-case rule, the acting-alone
  conditions (each one failing keeps the case for the doctor), and that a
  medication reply is never sent without approval.
- The doctor's page at 1100 and 390 px.

## 6. Constraints

gpt-4o-mini for the small calls; text sent to the model is de-identified as
today; one extra small call when a doctor sends a reply and none on the
patient's side; fail closed (no proposal rather than a guess).

## 7. Open questions for the team

1. The eight kinds in 4.2: are they the right ones?
2. Acting alone (4.6): build it now, off by default, or leave it out of the
   first version?

## 8. Calls made by Claude

Case memory rather than written rules; the kinds; three unchanged approvals
before acting alone, and never for medication; history counts as evidence but
not toward acting alone.

## 9. First version as built (2026-11-21)

### 9.1 The replay came first, and changed the plan

Section 5's replay was run on Dr. Okafor's 60 imported replies before anything
was built on it. No way of proposing her decision beat always naming her most
common one:

| Method | Right, of 60 |
|---|---|
| Always the most common decision (baseline) | 28 |
| Nearest past cases (the chosen approach of section 3) | 27 |
| Nearest past cases, the patient's chart added | 23 |
| A model shown her nearest cases | 21 |
| A model shown her nearest cases and the charts | 23 |
| A model alone | 18 |

Why: the history holds the same patient message, word for word, answered with
different decisions (the patients differ); the chart is a long problem list
that 60 cases cannot be learned from; and nine of ten replies are a first
response that asks the patient questions, so the decision was made before the
answers that decide it. The reader of decisions is not the cause: it agreed
with a plain wording rule on 56 of 60 replies.

The team then chose (2026-11-21) to make the first version collect what the
history lacks, and to propose nothing until the replay is clearly above its
baseline.

### 9.2 What was built

- **Kinds** (replaces 4.2). "come in" hid her judgement: three quarters of her
  replies ask for a visit, and what differs is how soon. One kind per reply,
  most urgent first: urgent, seen soon, book a visit, watch (self-care, a
  visit only if it does not get better), follow document, contact clinic, ask
  more (only asks), other; and two flags, "asks" and "medication". A safety
  net ("if it worsens, call 911") is not the decision.
- **Stored** (`case_decisions`, `decisions.record_decision`): when a doctor
  sends an answer, one small call reads the kind and one embedding is made.
  The row keeps the complaint, the interview's facts, whether the patient
  confirmed the summary, the reply, the kind and how the reply was made. It
  never blocks or fails the send; a reply that cannot be read is kept with no
  kind and is not evidence.
- **The doctor's word wins.** Every answered card shows "Your decision" as a
  list the doctor can change; a kind set by hand is marked and counts as
  evidence.
- **Similar past cases** (replaces 4.4). An open card lists this doctor's
  nearest past cases (at most 5, similarity 0.40 or more) with what was
  decided in each and the reply, under "A record, not a recommendation". No
  single decision is put forward. Cases are compared on the complaint alone:
  the interview's denials ("no shortness of breath") and the chart both
  pulled in the wrong cases.
- **History** (4.7): `scripts/decision_memory.py USERNAME --backfill` reads
  the doctor's earlier advice into rows marked "history".
- **The gauge**: the same script prints the replay with no model call, next to
  the baseline, and says "clearly above" only beyond two standard deviations
  of guessing at the baseline's rate. On the history: 30 of 60 against 28.
- **Not built**: a proposed decision, the draft given past cases as examples
  (4.4 step 3), and acting alone (4.6). They wait for the gauge.

Calls made by Claude: the split of "come in" by how soon; comparing on the
complaint alone; the floor of 0.40; the two-standard-deviation rule; kinds set
after sending, not before, to save a model call while the doctor types.

