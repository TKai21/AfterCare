# AfterCare - Multiple Doctors, Each With an AI Clone

Date: 2026-11-13
Status: draft, awaiting team approval
Builds on: `2026-11-13-doctor-clone-design.md` (implemented on branch `doctor-clone`). Where this document disagrees with that one, this one wins. It reverses two of its decisions: "one doctor" (now three) and "the AI never speaks in the doctor's first person" (now first person, always labelled as an AI clone).

## 1. Intent

The team wants each doctor to have their own AI clone, and patients to choose which doctor's clone to talk to. The clone answers as that doctor's AI clone, using that doctor's own past advice, and sends what it cannot answer to that doctor. Patients can also skip the clone and ask the real doctor directly.

Demo doctors (fictional, named by the team): **Dr. Ting Kai**, **Dr. Aaron Renny**, **Dr. Parin Patel**.

## 2. Decisions made with the team

- Three doctors, each with a doctor account and their own clone.
- **Clone voice: first person with a clear AI label (option A).** "I'm Dr. Ting Kai's AI clone. Based on my earlier advice, ..." Every clone answer carries a fixed "AI clone of Dr. X" label; replies the real doctor wrote look different.
- **One conversation per doctor (approach 1).** The Ask tab has one tab per doctor. Each doctor's clone has its own conversation history, uses only that doctor's advice, and sends unanswered questions only to that doctor. Approach 2 (one shared conversation with a doctor dropdown) was rejected: a clone would see another clone's answers as context, and patients could not tell who answered.

## 3. Goals

1. Three doctor accounts; patients see the list of doctors.
2. Ask tab: one tab per doctor, each with its own conversation and clone.
3. A clone uses only its own doctor's guidance entries and answers in that doctor's first person, labelled as an AI clone.
4. Questions are assigned to one doctor; each doctor's interface shows only questions assigned to them and only their own guidance.
5. Patients can ask the real doctor directly: an "Ask Dr. X directly" action under the clone conversation and a form in the Care team tab.
6. Reliability fix for the clone (section 8), found in the real-model check of the first version.

## 4. Non-goals and anti-goals

Non-goals: assigning each patient to one doctor (any patient may talk to any of the three); doctors seeing or editing each other's guidance; per-doctor specialties or availability; notifications.

Anti-goals (failure even if it works):

- **A patient believes a clone answer came from the human doctor.** Every clone answer shows the AI clone label; the clone never says or implies it is the human doctor ("I'm Dr. Ting Kai" alone is not allowed; "I'm Dr. Ting Kai's AI clone" is).
- **One doctor's advice appears under another doctor's name**, or in another doctor's conversation.
- **A question meant for one doctor reaches another doctor's queue.**
- Everything in the earlier spec's anti-goals still holds (applicability gate, no invented medical advice, document first, emergency backstop independent of model and memory).

## 5. Data model changes

- `chat_messages.doctor_id` (FK users, required for new rows): which doctor's conversation the message belongs to. Conversation history and `/api/conversation` are per (document, doctor).
- `care_team_questions.doctor_id` (FK users, nullable): the doctor the question is assigned to. Questions from the clone are assigned to that clone's doctor. Questions from a rejected extracted item (no doctor context) are unassigned and appear in every doctor's queue until one answers. (Designer's call.)
- `care_team_questions.source` gains `patient` (asked directly by the patient). The doctor's queue labels it "Asked by the patient".
- `doctor_guidance` is unchanged: entries already record `doctor_id`.
- Existing dev databases are recreated, as before.

## 6. Retrieval and the clone

- `guidance.retrieve` gains a `doctor_id` filter: a clone only ever sees entries whose `doctor_id` is its own doctor. The applicability gate, scope and document rules stay as they are.
- `/api/ask` takes `doctor_id` (required; 422 if it is not a doctor account). History, retrieval, the doctor name in the prompt, evidence titles, the not-covered copy and queue assignment all use that doctor.
- Prompt: replace the third-person rule with: "You are {DOCTOR}'s AI clone. Speak in the first person as {DOCTOR}'s AI clone ('my earlier advice', 'I advised'). Never say or imply you are the human doctor; when you introduce yourself, say you are {DOCTOR}'s AI clone." Evidence titles become "My earlier advice, <date>" in the model's context, and stay "Dr. X's advice (<date>)" on the patient's source chips.
- Everything else in the agent (authority order, number grounding, medication-change check, general-information check, emergency handling) is unchanged.
- The not-covered and fallback copy: "I couldn't answer this from your discharge document or my earlier advice, so I sent it to Dr. X." (first person, then the human doctor's name).

## 7. Patient experience

**Ask tab**

```
 Ask
 [ Dr. Ting Kai ] [ Dr. Aaron Renny ] [ Dr. Parin Patel ]
 ┌───────────────────────────────────────────────────┐
 │ AI clone of Dr. Ting Kai                          │
 │ Based on my earlier advice, you may take one      │
 │ extra 0.6 mg tablet after 1 hour [1].             │
 │ Sources: (1) Dr. Ting Kai's advice (Nov 12, 2026) │
 └───────────────────────────────────────────────────┘
 [ Ask Dr. Ting Kai's AI clone...          ] [ Send ]
 Want the real Dr. Ting Kai?  [ Ask Dr. Ting Kai directly ]
```

- The tab row lists all doctors (new patient endpoint `GET /api/doctors` -> `[{id, display_name}]`, ordered by id). The selected doctor is remembered per browser (local storage; first doctor by default).
- The lead text names the selected doctor: "Answers come from your discharge document and Dr. X's earlier advice, given by Dr. X's AI clone."
- Every clone answer has the "AI clone of Dr. X" label at the top (not only the first). Emergency answers keep the existing emergency banner and copy, unchanged.
- "Ask Dr. X directly" opens a small inline form, prefilled with the patient's last question in this conversation (editable), and sends it to that doctor's queue with source `patient`. A short confirmation line points to the Care team tab.

**Care team tab**

- A form at the top: "Ask a doctor directly", with a doctor select and a question box. Same endpoint as above.
- Each question shows which doctor it is for ("For Dr. X"). Replies keep the existing "Dr. X replied" block.

**Emergency on direct questions:** the emergency backstop runs on every direct question too. If it matches, the patient sees the existing emergency banner and copy, and the question is still saved for the doctor. (Designer's call: the doctor should still learn about it.)

New patient endpoint: `POST /api/care-team` body `{doctor_id, question}` -> the saved question view plus `{"emergency": {"terms": [...], "banner": bool}}`. 422 for an unknown doctor or an empty question (same length limit as `/api/ask`).

## 8. Reliability fix for the clone

The real-model check of the first version found the core demo question answered from the doctor's advice only 2 times in 3 with gpt-4o-mini. Two causes, two fixes:

1. **The model sometimes never calls `search_doctor_guidance`.** Fix: the server runs guidance retrieval for the patient's question before the first model call and puts the hits in the context as numbered evidence, labelled as this doctor's earlier advice (same evidence kind, numbering and grounding as tool results). The tool stays available for follow-up searches. If retrieval fails, the turn continues without it, as the tool already does.
2. **A reply with no JSON object at all used up the single repair round.** Fix: a reply with no JSON object gets one format-only retry ("reply with only the JSON object") that does not count as the repair round; content failures (citations, numbers, medication change) still get exactly one repair round. Emergency handling unchanged.

Acceptance: with the demo data, the colchicine question asked 10 times as demo2 to Dr. Ting Kai's clone is answered citing Dr. Ting Kai's advice at least 9 times; asked as demo (kidney stones) it never cites that advice.

## 9. Doctor experience

- The queue shows questions assigned to the logged-in doctor plus unassigned ones. Answering assigns an unassigned question to the answering doctor.
- "My guidance" shows only the doctor's own entries; PATCH and embed on another doctor's entry return 404.
- Citation counts count only the doctor's own entries (already true by id).
- The source label "Asked by the patient" for source `patient`.

## 10. Demo data

Seed three doctors: `dr.ting` / "Dr. Ting Kai", `dr.renny` / "Dr. Aaron Renny", `dr.patel` / "Dr. Parin Patel", all with password `doctor-password-1`. Keep patients `demo` (kidney stones) and `demo2` (gout). Each doctor gets at least one general entry that applies to one of the demo patients (passes the gate) and one open question in their queue; the existing Dr. Ting Kai entries stay. Answers are fictional, clinically plausible, with their numbers in the answer text, and marked in code as invented for the demo. README walkthrough updated: pick a doctor tab, ask, ask the real doctor directly, log in as that doctor and answer.

## 11. Testing

Unit and route tests in the existing style: a clone never retrieves another doctor's entries; history is per doctor; questions are assigned to the conversation's doctor and appear only in that doctor's queue (unassigned ones in all); `/api/ask` and `POST /api/care-team` reject unknown doctors; direct questions run the emergency backstop; prefetched guidance appears as evidence before any tool call; the format-only retry does not consume the repair round; prompt states the AI-clone rule. The frontend syntax test covers new scripts. A real-model run of the section 8 acceptance check before the demo, plus a browser check of the Ask tab, Care team tab and doctor page.

## 12. Left to the builder

File layout; exact wording of confirmation lines; the demo advice text for Dr. Aaron Renny and Dr. Parin Patel; how the doctor tab row looks on narrow screens (must not overflow).
