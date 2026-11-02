# Cross Artifact Consistency Check

| Check | Result | Evidence |
|---|---|---|
| Required file: 01_Project_Report/Project_Report.docx | PASS | Present and nonempty |
| Required file: 01_Project_Report/Project_Report.pdf | PASS | Present and nonempty |
| Required file: 02_Presentation/ML_Mini_Project_Presentation.pptx | PASS | Present and nonempty |
| Required file: 02_Presentation/Slide_Notes.md | PASS | Present and nonempty |
| Required file: 03_ML_Concepts/ML_Concepts_Explained.docx | PASS | Present and nonempty |
| Required file: 03_ML_Concepts/ML_Concepts_Explained.md | PASS | Present and nonempty |
| Required file: 04_Presentation_Script/Three_Member_Presentation_Script.docx | PASS | Present and nonempty |
| Required file: 04_Presentation_Script/Three_Member_Presentation_Script.md | PASS | Present and nonempty |
| Required file: 04_Presentation_Script/Member_1_Quick_Revision.md | PASS | Present and nonempty |
| Required file: 04_Presentation_Script/Member_2_Quick_Revision.md | PASS | Present and nonempty |
| Required file: 04_Presentation_Script/Member_3_Quick_Revision.md | PASS | Present and nonempty |
| Required file: 05_Experiments/Live_Experiments_and_Demonstrations.md | PASS | Present and nonempty |
| Required file: 05_Experiments/experiment_results.csv | PASS | Present and nonempty |
| Required file: 06_References/Verified_References.md | PASS | Present and nonempty |
| Required file: 06_References/Literature_Comparison.csv | PASS | Present and nonempty |
| DOCX parse: 01_Project_Report/Project_Report.docx | PASS | 68 top-level paragraphs |
| DOCX parse: 03_ML_Concepts/ML_Concepts_Explained.docx | PASS | 709 top-level paragraphs |
| DOCX parse: 04_Presentation_Script/Three_Member_Presentation_Script.docx | PASS | 291 top-level paragraphs |
| PDF parse: 01_Project_Report/Project_Report.pdf | PASS | 8 pages |
| PDF parse: 03_ML_Concepts/ML_Concepts_Explained.pdf | PASS | 23 pages |
| PDF parse: 04_Presentation_Script/Three_Member_Presentation_Script.pdf | PASS | 15 pages |
| Logistic Regression metrics | PASS | Matches saved JSON and synthetic held-out table |
| Decision Tree metrics | PASS | Matches saved JSON and synthetic held-out table |
| Random Forest metrics | PASS | Matches saved JSON and synthetic held-out table |
| Dataset sizes | PASS | 200 synthetic CSV rows; 40 duplicate rows removed; 160 unique examples (80 human-style, 80 AI-style); 20 synthetic problem groups; 96 training, 32 validation, 32 test samples. |
| Confusion matrix | PASS | Rows actual and columns predicted: [human,ai] |
| Handbook viva count | PASS | 99 distinct questions |
| Logistic numerical example | PASS | Recomputed sigmoid and weighted score independently |
| Manual metric example | PASS | Teaching counts: TP=8, TN=9, FP=3, FN=2 |
| TF-IDF numerical example | PASS | Smoothed IDF example independently recomputed |
| Gini numerical example | PASS | 3:1 node impurity independently recomputed |
| Member 1 revision sheet is one page | PASS | Word-exported PDF page count |
| Member 2 revision sheet is one page | PASS | Word-exported PDF page count |
| Member 3 revision sheet is one page | PASS | Word-exported PDF page count |
| Additional viva preparation | PASS | 30 general + 20 ML + 15 numerical + 15 implementation + 10 reliability |
| Additional question titles are unique | PASS | No repeated question titles within the 90-question preparation set |
| Speaker division | PASS | Six slides per member; 45-second slots, approximately 4.5 minutes each |
| PPTX slide count | PASS | 18 editable slides |
| PPTX speaker notes | PASS | All 18 slides contain notes |
| Model terminology: 01_Project_Report/Project_Report.md | PASS | All three implemented model names included |
| Scope disclosure: 01_Project_Report/Project_Report.md | PASS | Synthetic scope explicitly stated |
| Model terminology: 03_ML_Concepts/ML_Concepts_Explained.md | PASS | All three implemented model names included |
| Scope disclosure: 03_ML_Concepts/ML_Concepts_Explained.md | PASS | Synthetic scope explicitly stated |
| Model terminology: 04_Presentation_Script/Three_Member_Presentation_Script.md | PASS | All three implemented model names included |
| Scope disclosure: 04_Presentation_Script/Three_Member_Presentation_Script.md | PASS | Synthetic scope explicitly stated |

Research gaps are not counted as passed requirements: see Open_Information.md. Numeric results describe this synthetic fixture only.
