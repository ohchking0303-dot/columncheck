# ColumnCheck — local CSV import review

A small prototype for freelancers who review messy invoice exports before importing them into bookkeeping software. It proposes a column meaning with a genuinely trained local machine-learning model, then reports values that violate explicit format contracts. It never edits the input CSV.

**Status:** implemented and executed locally; not commercially validated or deployed as a hosted service. This is a review aid, not an accounting system. A human must check every suggested mapping.

## Run

Tested with Python 3.12.14 and the exact dependencies in `requirements.txt` (already installed in the build environment). For a fresh computer, create a virtual environment and install that requirements file from the official Python package registry. No model download, account, API key, hosted API or GPU is required.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python csv_review.py examples/invoices.csv --json-out review.json --html-out review.html
python -m unittest discover -s tests -v
python evaluate.py
```

Open `review.html` in a browser to inspect the escaped, offline report. Output files must not already exist, and the input and output paths must differ. Pick new paths when rerunning. The example is entirely synthetic and uses reserved `.test` email addresses. Existing execution evidence is in `evidence/`.

## What the ML actually does

At startup, scikit-learn trains a character 2–5-gram TF-IDF plus logistic-regression classifier on 72 original synthetic column headers, across email, date, money, identifier, quantity and description. It predicts meaning from the header only, not from cell values. This is traditional supervised machine learning, not a pretrained language model, generative AI, an LLM or a hand-written mapping disguised as a model. Learned model coefficients are fitted and used on every run.

The six-class classifier returns a proposed label, uncalibrated score and score margin. It abstains when the score is below 0.45, margin below 0.15, or the header has no known features. These thresholds are unvalidated engineering defaults, not a safety guarantee. Scores must not be interpreted as probabilities that the classification is correct. The training set is small and English-only.

Separate deterministic validators check nonempty values under the accepted proposed label:

- Email: a minimal syntax screen; no deliverability or complete RFC validation.
- Date: a real calendar date written exactly as YYYY-MM-DD.
- Money: a finite, plain decimal; no currency symbols, grouping, locale conversion or currency inference.
- Quantity: nonnegative whole numbers; negative returns and fractional units are flagged even when legitimate in another business workflow.
- Identifier/description: no semantic or format validation.
- Abstained columns: no format validation until a person resolves the mapping.

Missing values are listed separately, without assuming they are forbidden. Wrong model labels can produce wrong or missing validation warnings. Record numbers refer to CSV records (header = 1), not physical lines when quoted fields contain newlines. This prototype does not infer schemas from values, deduplicate data, validate invoice arithmetic or export repaired data.

## Measured synthetic evaluation

See `evidence/evaluation.json` for all 30 held-out predictions, including mistakes. Training and holdout have no identical normalized headers. They were authored for this project by the same assistant, share vocabulary, and are **not** an independently collected real-world benchmark. Three initially duplicated normalized holdout headers were caught by the overlap assertion and replaced before obtaining metrics. No hyperparameter or threshold tuning on holdout results was performed.

- Forced ML candidate: 26/30 correct.
- ML with abstention: 23/30 correct, 25/30 answered, 2 answered incorrectly.
- Simple token-rule baseline: 23/30 correct, 23/30 answered, none of the answered cases incorrect.

There is no demonstrated advantage over the baseline on the conservative decision metric. These numbers do not establish production accuracy, willingness to pay or contest eligibility. They show actual execution and expose a useful next decision: obtain independently labeled, authorized exports and determine whether ML adds value before pursuing a commercial product. Do not market this as an improvement over existing tools.

The invoice example detects four format conflicts in CSV record 3 and a missing email in record 4. The source SHA-256 in the report allows input integrity verification. Timing includes local training and inference, varies by machine, and is recorded rather than presented as a general performance claim.

## Privacy and safety boundaries

Application code makes no network calls. Dependency installation, if needed, uses the network; runtime analysis does not. Reports include column headers and record numbers but omit raw cell values. Headers can themselves be sensitive, and input hashes can identify known files: treat reports as private. There is no telemetry or arbitrary code/model deserialization. HTML escapes user-derived strings and has no remote assets or JavaScript. CSV formulas are never executed. Only UTF-8 comma-delimited input is supported, up to 5 MB, 10,000 data records and 100 columns; duplicate/blank headers and ragged records are rejected. Outputs are exclusive-create, and the source is read-only. This is a prototype, not a security-audited product.

## Plausible customer and business hypothesis

A bookkeeping freelancer repeatedly importing client exports might pay for reliable mapping review and auditable issue reports. A one-time local-tool license or paid setup service could be tested. No customer interviews, purchases, paid users or willingness-to-pay evidence have been obtained. This remains a hypothesis. Before charging: independently evaluate real files with permission, extend locale contracts, strengthen abstention, measure time saved, and test packaging on customer machines.

## Open-source and model provenance

All project source and synthetic examples were newly authored for this prototype on 2026-10-18. No previous customer work or existing demo was reused. The source and synthetic fixtures are offered under MIT; see `LICENSE`. The trained model is reproducible from the included training records and code. No external model weights are redistributed. Dependencies retain their own licenses; see `THIRD_PARTY_NOTICES.md`.

The open implementation allows offline inspection, replacement of the classifier, and extension of validators. These are architectural benefits, not measured superiority over closed alternatives. AI assistance was used to write the code, tests and documentation; the running product also executes a learned model, separately from that development assistance.

A pretrained alternative was considered: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), whose official card lists Apache-2.0 and 22.7M parameters. The build environment had no PyTorch, Transformers, Sentence Transformers or ONNX Runtime. Using it would require additional packages and weight downloads, so it was not installed or executed. We make no claims about its behavior here.

## Version 2: human mapping overrides

Use `--schema examples/schema.json` to specify known column meanings. The JSON object maps exact headers to email, date, money, identifier, quantity, description or review_required. Unknown headers, duplicate JSON keys, unsupported types and output/schema path collisions are rejected. Each column retains the ML suggestion alongside effective_label and label_source; user overrides do not silently replace model evidence. Setting review_required suppresses format validation. This is explicit human control, not a measured model improvement.

Run: `python csv_review.py examples/invoices.csv --schema examples/schema.json --json-out new-review.json --html-out new-review.html`. Version 2 passed 35 tests; see evidence/tests_v2.txt. The 30-header synthetic evaluation remains unchanged. A 40-second illustrated walkthrough of actual execution results is included in demo_video/columncheck_demo.mp4. It is not a live screen recording.

A local HTML screenshot attempt failed because the execution environment could not create the browser process socket. HTML visual QA remains incomplete; no screenshot or browser-rendering success is claimed.
