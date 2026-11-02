# YASH RAJ AGARWAL Quick Revision

Slides 1–6; target approximately 4.5 minutes.

Motivation; objective; four verified sources and literature gap; actual architecture; synthetic data and grouping.

200 synthetic CSV rows; 40 duplicate rows removed; 160 unique examples (80 human-style, 80 AI-style); 20 synthetic problem groups; 96 training, 32 validation, 32 test samples.

Models: Logistic Regression, Decision Tree, Random Forest. Selected: Logistic Regression. Test metrics: all four 1.000; confusion matrix [[16,0],[0,16]]. Both labels are synthetic style categories written into a generator, not verified human or LLM provenance. The 20 group IDs vary a few filtering templates and do not represent 20 independent algorithmic tasks. Perfect scores describe template separation only.

## What does this project actually classify?

Synthetic code-style labels. The generator assigns human and ai strings to templates. Neither label proves actual provenance.

## How many records are used?

160 unique examples from 200 rows. Forty exact duplicates are removed after stripping boundary whitespace.

## What is group splitting?

Keeping related records in one partition. GroupShuffleSplit keeps each problem_id exclusively in training, validation or test.

## Are there 20 independent tasks?

No. The IDs reuse a few filter templates with naming and modulus variants.

## Demonstration recovery

Start backend with .venv\Scripts\python.exe -m uvicorn backend.main:app and frontend with npm run dev --prefix frontend. If startup fails, show the captured real screenshots and experiment CSV. Never invent a live result.
