# What leaves this machine

AfterCare sends text to a language model to answer patients and to help
doctors. This page states what is sent, in what form, and what is not
protected. The code is `backend/privacy.py`; the design is
`docs/superpowers/specs/2026-11-18-deidentification-design.md`.

## The rule

Every call to the model goes through one wrapper. Before a call, anything that
identifies a person or a place is replaced by a placeholder; after it, the
placeholders are put back. Patients and doctors see the original words, and the
database stores the original words. Only the copy that leaves is changed.

A patient's document as the model sees it:

```
[FACILITY_1] - Discharge Summary
Patient: [PATIENT_1] (fictional), 52-year-old adult
Attending: Dr. [DOCTOR_1], Family Medicine
Colchicine 0.6 mg by mouth twice daily for 10 days.
See Dr. [DOCTOR_1] on November 24, 2026 for a pain check.
Call the clinic at [PHONE_1] if you have diarrhea.
```

If a known name is still in a call after masking, the call is not sent. The
emergency check runs on this machine and does not depend on the model.

## What is sent

| Data | Leaves the machine | In what form |
|---|---|---|
| Patient, doctor and facility names | No | Replaced by `[PATIENT_n]`, `[DOCTOR_n]`, `[FACILITY_n]` |
| Phone numbers, e-mail addresses, street addresses | No | Replaced by placeholders |
| Record numbers and dates of birth on labelled lines, SSN-shaped numbers | No | Replaced by placeholders |
| Symptoms, diagnoses, drugs, doses | Yes | As written; the model needs them |
| Dates (discharge, follow-up, how long a symptom has lasted) | Yes | As written |
| Age and sex | Yes | As written |
| The doctor's earlier advice and habits | Yes | With names and numbers masked as above |
| Dictation recordings | Yes | The audio itself; it is not stored by AfterCare |
| Passwords, login tokens | No | Never part of a model call |

## Where names come from

1. Account names of patients and doctors.
2. Labelled lines in a discharge document: `Patient:`, `Name:`, `DOB:`, `MRN:`,
   `Address:`, `Attending:`, `Physician:`, `Provider:`, and the facility line.
3. "Dr. Firstname Lastname" anywhere in a text.
4. Patterns: phone numbers, e-mail addresses, street addresses.

## Known gaps

- **Dates are kept.** They are needed to answer. This is weaker than the HIPAA
  Safe Harbor list, which removes dates.
- **A name typed in free text** that appears in none of the sources above (for
  example "my wife Maria drove me") is sent as written.
- **Voice.** A recording cannot be de-identified. The page says so while
  recording.
- **Rare conditions.** A diagnosis and medication list can identify a person
  even without a name.
- **Maintenance scripts.** The scripts under `scripts/` that call the model
  directly (seeding the demo, building the library, evaluations) do not go
  through the wrapper. They work on demo and public data.
- This is a class project. It has not been reviewed for HIPAA compliance.

## Turning it off

`PRIVACY_MASKING=0` sends text as written. It is meant for a model that runs
on a machine we control (`LLM_API_BASE` pointing at it), where nothing leaves.

## How to check it

`tests/test_privacy.py::test_upload_and_ask_send_no_identifier` uploads a
document, asks a question that names the patient, and fails if any call to the
model contains the patient's name, the doctor's name, the facility, the clinic
phone or the record number.
