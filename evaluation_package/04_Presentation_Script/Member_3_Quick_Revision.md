# AMAN SHARMA Quick Revision

Slides 13–18; target approximately 4.5 minutes.

API request flow; real screenshots; eight experiments; outputs; reliability limits; future data protocol.

200 synthetic CSV rows; 40 duplicate rows removed; 160 unique examples (80 human-style, 80 AI-style); 20 synthetic problem groups; 96 training, 32 validation, 32 test samples.

Models: Logistic Regression, Decision Tree, Random Forest. Selected: Logistic Regression. Test metrics: all four 1.000; confusion matrix [[16,0],[0,16]]. Both labels are synthetic style categories written into a generator, not verified human or LLM provenance. The 20 group IDs vary a few filtering templates and do not represent 20 independent algorithmic tasks. Perfect scores describe template separation only.

## What happens on Analyze?

React posts JSON, FastAPI validates, saved inference returns JSON. The fitted transformer and classifier run; training and source execution do not.

## What is feature ablation?

Removing feature groups and retraining. Validation comparisons reveal conditional reliance, though this trivial fixture gives identical class metrics.

## Does the factorial probe prove generalization?

No. It is one unlabeled unseen task, so accuracy cannot be computed.

## What if a human writes a docstring?

The model may raise the AI score. Docstrings are a synthetic class cue here; this is an example of why false accusations are possible.

## Demonstration recovery

Start backend with .venv\Scripts\python.exe -m uvicorn backend.main:app and frontend with npm run dev --prefix frontend. If startup fails, show the captured real screenshots and experiment CSV. Never invent a live result.
