# De-identification before the model - design

Date: 2026-11-18. Status: draft, waiting for the team's approval.

## 1. Intent

Patient privacy. Today every model call sends the patient's words and the whole
discharge document (name, phone numbers, clinic, clinicians, dates, diagnosis)
to the model provider. The team wants the patient's identity to stay on our
machine, without giving up answer quality and without new running cost.

## 2. Goals, non-goals, anti-goals

Goals
- Nothing that identifies the patient, a clinician or the facility reaches the
  model provider: they are replaced by placeholders before a call and put back
  after it, so patients and doctors see exactly what they see today.
- The guarantee is enforced by code, not by trust: a call whose payload still
  contains a known identifier is not sent.
- One page that states what leaves the machine and in what form.

Non-goals
- A local model (a setting change later; this design does not block it).
- Database encryption, access logs, consent flows, HIPAA compliance.
- Hiding the medical content (diagnosis, drugs, symptoms): the model needs it.

Anti-goals (a failure even if it works)
- A drug name, dose, number or symptom is masked, so an answer gets worse.
- A placeholder such as `[PHONE_1]` is shown to a patient or a doctor.
- Stored data is altered: the database keeps the original text.

## 3. Approaches considered

- A: local model for everything. Best privacy, but this machine (16 GB) runs
  only small models, weaker than the current one at reading the document and
  judging danger signs.
- **B, chosen: de-identify at the boundary.** Same model and quality, no extra
  model calls, and the guarantee is checkable.
- C: keep sending everything and rely on the provider's terms. No code, no
  guarantee we can test.

A can be added on top of B later by changing `LLM_API_BASE`.

## 4. Design

### 4.1 One boundary

Every outbound call already goes through `LLMClient` (`chat`, `embed`,
`transcribe`). A wrapper, `PrivateLLM(inner, vault)`, has the same interface
and is what the app uses:

- `chat`: masks every message's text (and tool results) on the way out; on the
  way back unmasks the reply's content and its tool-call arguments.
- `embed`: masks the texts. Vectors come back unchanged.
- `transcribe`: passed through (4.5).

The rest of the code never sees a placeholder, so validation, number
grounding, the emergency backstop and storage work on the original text.

### 4.2 What is masked

An identifier is a (kind, value) pair; each value gets a stable placeholder per
request, `[PATIENT]`, `[DOCTOR_1]`, `[FACILITY_1]`, `[PHONE_1]`, `[EMAIL_1]`,
`[ID_1]`, `[ADDRESS_1]`.

Sources, all computed locally:
1. **Known values**: the patient's and every doctor's account name
   (`display_name`, `username` when it is a real name), with and without "Dr.".
2. **Document header lines**, read by rules when a document is uploaded and
   stored with it: `Patient:`, `Name:`, `DOB:`, `MRN:`, `Address:`, `Phone:`,
   `Attending:` / `Physician:` / `Provider:`, and the facility line (the first
   line that ends in "Hospital", "Medical Center", "Clinic", ...). Also every
   "Dr. Firstname Lastname" in the text.
3. **Patterns** in any outbound text: phone numbers, e-mail addresses, URLs
   with a host, SSN-shaped and MRN-labelled numbers, street addresses.

Matching is case-insensitive on word boundaries, longest value first; a full
name also masks its parts ("Sam Delgado", "Delgado", "Sam") when the part is
at least 3 letters and is not a word of the drug lexicon or a common English
word list bundled with the app.

Not masked: drug names, doses, any number that is not one of the patterns
above, symptoms, diagnoses, and **dates** (see section 7).

### 4.3 The guard

After masking, the wrapper checks the payload: if any identifier value of this
request still appears, the call is not sent and raises `LLMError`. Callers
already treat that as a failed model call (the emergency backstop is local and
still answers). A reply that contains a placeholder the vault does not know is
treated as a failed call, so no placeholder reaches a person.

### 4.4 Stored vectors

Existing vectors were computed from unmasked text. The index records the
embedding model name; the wrapper reports a different name
(`<model>+masked`), so the existing "model changed, re-index" path rebuilds
them from masked text.

### 4.5 Dictation

Audio cannot be de-identified. The recording is sent to the transcription
model as today and is not stored. The Dictate button gets a short line: "Your
recording is sent to a speech service to be turned into text and is not
kept." A local speech model is a separate, later change.

### 4.6 The statement

`docs/privacy.md`: a table of each kind of data, whether it leaves the
machine, and in what form; the known gaps (section 7); how to verify it (the
test in section 5).

## 5. Verification

- Unit tests of masking and unmasking (round trip is the identity; placeholders
  are stable within a request; drug names and doses survive).
- A recording fake model in the route tests: across an ask, an interview, a
  doctor draft, an upload and a guidance search, no payload contains the
  patient's name, the doctor's name, the clinic phone or the facility name.
- One real run on a copy of the demo database, comparing answers to today's
  on the questions used on 2026-11-18 (naproxen, colchicine, water, walking,
  diarrhea, follow-up date, runny nose interview).

## 6. Constraints

No new model calls and no new paid service. Repo docs in English, no em dash,
never write to `aftercare-new.sqlite3` from tests. The confidence threshold
was calibrated on unmasked prompts; if the comparison run in section 5 shows
lower scores, it is recalibrated.

## 7. Known gaps (stated in docs/privacy.md)

- **Dates are kept.** Follow-up dates and day counts are needed to answer, and
  a date without a name identifies little. This is weaker than the HIPAA Safe
  Harbor list, which removes dates.
- **Names typed in free text** that are in no source of 4.2 (for example "my
  wife Maria drove me") are not masked. A local name recogniser can be added
  if the team wants this closed.
- **Voice** (4.5).
- The diagnosis and medication list themselves can identify a person with a
  rare condition.

## 8. Left to the builder

The exact patterns, the common-word list, where the document's identifiers are
stored (a JSON column on the document is enough), and the wording of
`docs/privacy.md`.

## 9. Decided by the team

De-identify at the boundary rather than switch to a local model (2026-11-18).

## 10. Calls made by Claude

Keep dates; mask clinician and facility names as well as the patient's; fail
closed on a leftover identifier; no name recogniser in the first version;
dictation keeps the cloud speech model with a notice.
