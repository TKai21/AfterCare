# AfterCare - Drug Labels From Free APIs, Approved by a Doctor

Date: 2026-11-13
Status: draft, awaiting team approval
Builds on: the education library (`backend/knowledge.py`), the doctor interface and AI clones (`2026-11-13-doctor-clone-design.md`, `2026-11-13-multi-doctor-clones-design.md`).

## 1. Intent

The team wanted the clone's general drug information to come from an authoritative reference. UpToDate and Micromedex cannot be used: the USC subscription allows reading them, but their terms forbid copying their text into another system, and their APIs are sold only to hospitals and EHR vendors. FDA drug labels are public and have free APIs, so the library gets its drug information from them.

Today the library is a hand-maintained list (`knowledge/manifest.json`, 30 sources). It has one drug label (ibuprofen) and no gout sources, although demo patient `demo2` has gout. Any new discharge document with a new drug has no library coverage. This feature fills that gap automatically, while keeping the library's rule that no text reaches a patient before a person has reviewed it.

## 2. Decisions made with the team

- **Approach C: fetch automatically, a doctor approves before use.** When a patient confirms a discharge document, the server looks up each medication's FDA label in the background. The label goes into a "Drug labels to review" list on the doctor page. Only approved labels are searchable.
- Rejected: A (a command that adds labels to the manifest ahead of time; only prepared drugs are covered) and B (index fetched labels straight away; a wrong product would reach patients unreviewed).
- The team accepted these designer calls in the playback:
  - One review per drug, shared by all patients.
  - Any doctor may approve.
  - A label is used only for patients whose document lists that drug.
  - Only the drug name leaves the server.
  - A label never justifies a medication change.
  - API failures never block the patient.
  - Disease pages (MedlinePlus) are added to the manifest by hand, not fetched automatically.
  - RxNorm is used only for labels for now.

## 3. Goals

1. On confirmation, each medication item is matched to an RxNorm ingredient and to DailyMed label candidates, without slowing the confirm request.
2. Doctors see pending labels, read the extracted text, approve, reject, or pick another candidate.
3. An approved label is added to the library at once, without a full rebuild. It survives later full rebuilds.
4. Label passages are retrieved only for patients whose confirmed document has that ingredient.
5. Gout patient-education pages are added to the manifest, so `demo2` has library coverage for the condition.

## 4. Non-goals and anti-goals

Non-goals:
- Automatic disease pages.
- Using RxNorm to normalise guidance tags.
- Drug-interaction checking (the NLM interaction API was retired in 2024; interactions appear only as label text).
- UpToDate or Micromedex in any form.
- Refreshing approved labels when the FDA updates them.
- Patient-facing UI changes beyond the source chip.

Anti-goals (failure even if it works):
- **Label text a doctor has not approved reaches a patient,** through answers, sources or search.
- **A patient gets passages from a label for a drug that is not on their document.**
- **A label of a different drug, or a non-matching product, is approved without the doctor seeing what it is.** The review card must show the ingredient, the product title and the source document line.
- **Anything besides a drug name is sent to an outside service.** Patient name, document text and ids stay on the server.
- **The confirm step fails or waits because RxNorm or DailyMed is slow or down.**
- A label passage is used to justify changing, stopping or adding a medication. The existing MEDICATION_CHANGE rule already forbids this; it must still hold.

## 5. Outside services (free, no key)

| Service | Use | Example |
|---|---|---|
| RxNorm (rxnav.nlm.nih.gov) | Medication name to ingredient RxCUI and name | `/REST/approximateTerm.json?term=Colcrys 0.6 mg`, then `/REST/rxcui/{id}/related.json?tty=IN` |
| DailyMed (dailymed.nlm.nih.gov) | Label candidates for an RxCUI, then label XML by setid | `/dailymed/services/v2/spls.json?rxcui=2683`, then the existing `DAILYMED_SPL_XML` |

- The server sends only the medication name text and the RxCUI.
- Requests have timeouts and a modest rate. RxNav asks for at most 20 requests per second.
- All calls go through one injectable `httpx` transport, so tests and the demo seed run offline.
- Section extraction reuses `dailymed_sections`. The sections to take are the patient-useful ones when present:
  - Boxed warning, Dosage and administration, Contraindications.
  - Warnings and precautions (or Warnings), Drug interactions, Adverse reactions.
  - Patient counseling information or Medication Guide.
- The exact LOINC code list is left to the builder. It must include dosage, warnings and interactions.

## 6. Data model (app database)

`drug_labels`, one row per ingredient:
- `rxcui` (unique) and `ingredient`.
- `status`: `pending`, `approved`, `rejected`, `no_label` or `failed`.
- `candidates` (JSON list of up to 5 `{setid, title, published}`, newest first) and the chosen `setid`, `title`, `url`.
- `text`, `sha256`, `collected_at`, `error`.
- `reviewed_by`, `reviewed_at`.

`document_drugs`, one row per confirmed medication item:
- `document_id`, `extracted_item_id`, `name` (as extracted).
- `rxcui` (nullable) and `status`: `resolved`, `unresolved` or `failed`.

Rules:
- A new ingredient creates a `pending` label from the newest candidate. An ingredient that already has a row (any status) is not re-queued.
- Reject keeps the row as `rejected`, so the drug is not fetched again. A doctor can still pick another candidate from it, which fetches that label and sets `pending` again.
- `failed` (network or parse error) and `unresolved` names show a "Try again" action on the doctor page.
- Existing dev databases are recreated, as before.

## 7. Library integration

- On approve, the server calls `KnowledgeStore.add` with the id `dailymed-rxcui-<rxcui>`, title `DailyMed label: <ingredient> (<product title>)`, topic `Medications`, the DailyMed `drugInfo.cfm?setid=` URL and the label's `rxcui`.
  - If embedding fails, the approval is not saved and the doctor sees an error.
- The library `documents` table gains a nullable `rxcui` column. Manifest sources have none and stay visible to every patient, as today.
- `KnowledgeStore.retrieve` takes the patient's set of RxCUIs. It drops label passages whose `rxcui` is not in the set. The ask route builds the set from `document_drugs` for the patient's active confirmed document.
- Full rebuilds (`manage_knowledge.py index`) include approved `drug_labels` rows alongside reviewed manifest snapshots, so an approved label is never lost.
- Un-approving an approved label is out of scope. A doctor can reject before approval only.

## 8. Doctor experience

A "Drug labels to review" section on the doctor page:
- One card per pending label: ingredient, product title, the source medication line and patient document (username only), and a collapsible view of the extracted text.
- Actions: Approve, Reject, and "Pick another label" (the other candidates by title and date).
- `failed` and `unresolved` entries appear in the same section with the error, the medication name and "Try again".
- Section empty: one quiet line, "No drug labels waiting."
- Endpoints, doctor-only:
  - `GET /api/doctor/labels`
  - `POST /api/doctor/labels/{id}/approve`
  - `POST /api/doctor/labels/{id}/reject`
  - `POST /api/doctor/labels/{id}/candidate` with `{setid}`
  - `POST /api/doctor/labels/{id}/retry` and `POST /api/doctor/document-drugs/{id}/retry`
- Concurrent approve and reject: the first one wins, the other gets 409. This is the same conditional-update pattern the question queue uses.

## 9. Patient experience

No new screens. When an answer cites an approved label, the source chip reads "DailyMed label: colchicine (...)" and links to the DailyMed page. General-information answers (rule 7, e.g. a missed dose) can now cite label text. All agent rules are unchanged.

## 10. Hand-added sources

Add MedlinePlus "Gout" (`https://medlineplus.gov/gout.html`) and NIAMS "Gout" (`https://www.niams.nih.gov/health-topics/gout`) to the manifest. Add `www.niams.nih.gov` to `ALLOWED_HOSTS`. Collect, review and index them through the existing commands.

## 11. Demo data

The seed must work offline. It records the demo patients' medications in `document_drugs` using fixture responses, and leaves their labels `pending`. The walkthrough then shows: log in as `dr.ting`, approve the colchicine label, log in as `demo2` and ask a missed-dose question.

README: a short "Drug labels" section explains the flow. It also explains why UpToDate and Micromedex are not used (licence terms).

## 12. Testing

Unit and route tests in the existing style, using an `httpx.MockTransport`. No real network in tests. Cover:
- Name-to-RxCUI resolution, including no match and service errors.
- Candidate selection and section extraction from a fixture label.
- Confirm returns before lookups finish, and succeeds when the services are down.
- One label row per ingredient across patients.
- Pending and rejected labels are never retrieved.
- Approval adds the label to the store; an embedding failure leaves it pending.
- Retrieval drops labels for RxCUIs not on the patient's document, and keeps manifest sources.
- A full rebuild keeps approved labels.
- 409 on a concurrent review.
- Non-doctors get 403.
- Only the name is sent outside (assert on the recorded requests).

The frontend syntax test covers new scripts. Before the demo, the team does one real-network run (confirm `demo2`, approve, ask) and a browser check of the new doctor section.

## 13. Settled after the builder check (designer's calls)

1. **Patient filter placement.**
   - The ask route passes the patient's RxCUI set into the library search tool, the same way it passes the document today.
   - `retrieve` filters out other label rows in SQL, before ranking and the top-k cut, so label passages never crowd out manifest sources.
   - If the patient has no confirmed document, the set is empty.
2. **Lookup term and re-confirmation.**
   - Only items with `category == "medication"` are looked up.
   - The term is `details.name` plus `details.dose` when present (for example "colchicine 0.6 mg"). If that finds nothing, `details.name` alone is tried.
   - `document_drugs` rows belong to one document. A new document gets its own rows, and the patient set always comes from the active confirmed document.
3. **Combination products.**
   - `document_drugs` has one row per (item, ingredient), so a combination pill gives several rows.
   - Each ingredient gets its own label row. The patient set is every ingredient.
4. **Candidate choice.** A real check (2026-11-13) found 67 colchicine labels for the ingredient RxCUI. Most were repackagers, and one was LODOCO, a 0.5 mg colchicine product for a different indication. Rules:
   - When the term resolves to a strength-level RxCUI (`colchicine 0.6 MG`, 315714), candidates come from that RxCUI. Otherwise they come from the ingredient RxCUI.
   - Exclude titles that name another active ingredient.
   - Order newest first and keep 5.
   - The card shows the labeler, the dose form and the published date, so a wrong product is visible.
   - Ingredient-level queries do return labels (verified), so they are a valid fallback.
5. **Library schema upgrade.** `KnowledgeStore.__init__` adds the nullable `rxcui` column with `ALTER TABLE` when an existing index lacks it. No forced rebuild is needed.
6. **Number grounding.** Label passages are library evidence and go through the same `check_text` rule as every other library passage.
7. **Failures and restarts.**
   - There is no automatic retry. Rows have an extra transient status `fetching`.
   - On startup, rows left in `fetching` become `failed` with the error "Interrupted, try again". The doctor's "Try again" restarts them.
8. **Who sees pending labels.** Every doctor sees every label. Patients are not assigned to doctors (multi-doctor spec, non-goals), and the label library is shared.

## 14. Left to the builder

- File layout (suggested: `backend/druglabels.py` and routes in `backend/routes_doctor.py`).
- Background mechanism (FastAPI background task or an in-process task; it must use its own DB session).
- The exact LOINC section list.
- How candidates are ranked beyond "newest first". Prefer labels whose title matches the extracted dose form when one does.
- The wording of card text and errors.
