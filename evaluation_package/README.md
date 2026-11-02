# CodeTrace AI Evaluation Package

This package documents the actual schema-2 implementation and measured outputs. It includes an IEEE-template report, editable 18-slide PowerPoint, repository-specific ML handbook, three-member script and revision sheets, real application screenshots, reproducible experiments and a bounded verified-source record.

## Start here

- 01_Project_Report/Project_Report.docx and Project_Report.pdf: report built from the supplied official IEEE DOCX, with academic cover and full-width evidence appendix.
- 02_Presentation/ML_Mini_Project_Presentation.pptx: 18 editable slides with speaker notes, editable architecture, tables and a native chart. Slide_Notes.md contains the exact verbal script.
- 03_ML_Concepts/ML_Concepts_Explained.docx and .md: 12 chapters, the 18-feature catalog, worked mathematics, actual source excerpts, logical walkthrough and 99 viva questions in nine categories. A PDF reading copy is also included.
- 04_Presentation_Script/: balanced six-slide sections for Yash, Nishaanth and Aman, transitions, pronunciation, 90 additional preparation questions and three quick revision sheets. DOCX, Markdown and PDF reading copies are included.
- 05_Experiments/: the runner, eight experiment types, actual output CSVs, paired source/configuration comparisons and charts.
- 06_References/: publisher-verified bibliography, comparison CSV, institutional and logo sources, indexing/access limitations.
- Technical_Inventory.md: exact implementation/version inventory and corrections.
- Consistency_Check.md / .json: cross-artifact checks and explicit remaining information gaps.

## Reproduce the application and experiments

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
.\.venv\Scripts\python.exe backend\train.py
.\.venv\Scripts\python.exe -m unittest backend.test_pipeline -v
.\.venv\Scripts\python.exe -m pip install -r evaluation_package\05_Experiments\requirements.txt
.\.venv\Scripts\python.exe evaluation_package\05_Experiments\run_experiments.py
.\.venv\Scripts\python.exe evaluation_package\05_Experiments\make_charts.py
.\.venv\Scripts\python.exe -m uvicorn backend.main:app
```

In a second terminal: `npm install --prefix frontend` then `npm run dev --prefix frontend`. Open http://127.0.0.1:5173/ . The project README explains initial virtual-environment setup. Run plotting after experiments so charts reflect the latest CSV. Experimental configurations never replace the deployed model. A deliberate change to features or data requires retraining and rebuilding the package so all artifacts remain consistent.

## Defensible interpretation

200 CSV rows become 160 unique synthetic examples, balanced at 80 per style label. The split is 96/32/32 across disjoint IDs, but a few template families are shared. All three classifiers score 1.000 on the artificial test partition; selected Logistic Regression has confusion matrix [[16,0],[0,16]]. These are genuine measured outputs and are not evidence of real-world authorship detection. The unseen factorial probe has no independently verified label.

## Required before submission

Replace [INSERT FACULTY GUIDE NAME] after confirming the guide. The requested 20–25 indexed journal articles could not be fully verified: four publisher-verified journal papers are included with explicit coverage/indexing gaps, rather than invented sources. See Open_Information.md. The institutional name and logo source are verified; the logo is MAHE's authentic institutional asset.

## Checks completed

Eight meaningful backend regression tests; actual HTTP valid/invalid input checks; frontend production build; full npm dependency audit (zero known vulnerabilities at verification); Word DOCX opening/PDF export; PowerPoint opening of all 18 slides; package/layout/font/chart-data validation; PDF/page and slide visual review; model/data/metric consistency checks. Rehearsal timing and real-world detection performance remain unvalidated.

The .build directory contains reproducible authoring scripts and private QA intermediates; it is excluded from the delivery ZIP. Browser screenshots are real localhost captures. Services used for verification are stopped after preparation.
