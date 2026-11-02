# NISHAANTH S Quick Revision

Slides 7–12; target approximately 4.5 minutes.

18 features; TF-IDF; scaling; three models; validation selection; metrics; global explanation limits.

200 synthetic CSV rows; 40 duplicate rows removed; 160 unique examples (80 human-style, 80 AI-style); 20 synthetic problem groups; 96 training, 32 validation, 32 test samples.

Models: Logistic Regression, Decision Tree, Random Forest. Selected: Logistic Regression. Test metrics: all four 1.000; confusion matrix [[16,0],[0,16]]. Both labels are synthetic style categories written into a generator, not verified human or LLM provenance. The 20 group IDs vary a few filtering templates and do not represent 20 independent algorithmic tasks. Perfect scores describe template separation only.

## How does tokenization work?

Python tokenize yields lexical tokens. The numeric token statistics include names, operators and literals while excluding comments and layout.

## Does nesting depth mean raw AST depth?

No. It increments selected control constructs and function definitions; expression layers alone do not increment it.

## What is the smoothed IDF formula?

1+ln((1+N)/(1+df)). The extra ones avoid zero divisions and follow TfidfVectorizer defaults.

## Why did you choose XGBoost?

We did not implement it. It belongs to the synopsis and future scope; the actual project uses three scikit-learn baselines.

## Demonstration recovery

Start backend with .venv\Scripts\python.exe -m uvicorn backend.main:app and frontend with npm run dev --prefix frontend. If startup fails, show the captured real screenshots and experiment CSV. Never invent a live result.
