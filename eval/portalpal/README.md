# PortalPal evaluation data

`syppm.csv` is the PortalPal patient message response drafting dataset,
downloaded once from
<https://huggingface.co/datasets/PortalPal-AI/Patient-Message-Response-Drafting/resolve/main/syppm_huggingface.csv>
(SHA-256 `b595c00d47aac3739a541ae2f3efaa975fa28a9cb2a9f5c3a28692e3927196a2`) and
committed unchanged. It has 100 rows: a semi-synthetic patient message, a real
de-identified EHR summary, and a reply written by one MD with 15+ years of
experience in primary care. The messages are synthetic; the EHR elements were
approved for public release by the authors' IRB and collaborating hospital
(see the dataset card).

## Licence

CC BY-NC 4.0 (<https://creativecommons.org/licenses/by-nc/4.0/>): attribution,
non-commercial use only. This repository uses it for a class project
evaluation, not commercially.

## Citation

```
@misc{seegmiller2026clinicianediteval,
      title={How Much Would a Clinician Edit This Draft? Evaluating LLM Alignment for Patient Message Response Drafting},
      author={Parker Seegmiller and Joseph Gatto and Sarah E. Greer and Ganza Belise Isingizwe and Rohan Ray and Timothy E. Burdick and Sarah Masud Preum},
      year={2026},
      eprint={2601.11344},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2601.11344},
}
```

## What this repository adds or changes

The CSV itself is not modified. Around it:

- **Ids and split** (`split.json`, `scripts/portalpal_data.py`): each row is
  named by its index (`pp-000` to `pp-099`) and the rows are split once with
  seed 542 into 60 memory rows (the physician's replies become the evaluation
  doctor's saved answers), 15 tuning rows and 25 test rows.
- **EHR summary to document** (`ehr_document`): each EHR summary is the text
  of that patient's confirmed discharge document, with one item per active
  problem (category `other`, "Active problem: ...") and one medication item
  per active medication line. Recently ended medications, past diagnoses and
  demographics stay in the text only. "*No Active Medications*" gives no
  medication item.
- **Scripted replies** (`replies.json`, written once by
  `scripts/eval_portalpal.py --write-replies`, then read and fixed by hand):
  for each tuning and test row, three patient replies to the clone's
  questions: benign, warning (one sign the physician's reply names, given as
  `warning_sign`) and warning paraphrase (the same sign without its key
  words, given as `warning_paraphrase_of`). These are new test data, not part
  of the dataset.
- The evaluation doctor is shown as "Portal physician", never a real name.

## Where the data goes

The rows are used only inside temporary databases that each run creates and
deletes (`scripts/eval_portalpal.py`); nothing from the dataset is written to
the application's configured database. Results go to `results/<split>/`.
