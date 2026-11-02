# Slide Notes

Timing target: approximately 13.5 minutes, including short diagram pauses and demonstration actions. Each member presents six slides and approximately 4.5 minutes. Rehearse at 130–145 words per minute; timing is an estimate, not a measured rehearsal.

## Slide 1 CodeTrace AI

Speaker: YASH RAJ AGARWAL. Approximate slot: 45 seconds including visual cues.

Cue: Keep eye contact; use the slide as a visual anchor.

Good morning. We are Yash Raj Agarwal, Nishaanth S and Aman Sharma. Our project is titled Stylometric and Structural Classification Framework for Detecting LLM-Generated Code Submissions. We built CodeTrace AI as a local machine learning demonstration. It extracts features from Python source without running the program, then uses a saved classifier to estimate which labeled style category the sample resembles. We will explain the implementation, show its measured results, and demonstrate where it is fragile. We want to be clear from the beginning: the included data is synthetic, so this presentation does not claim validated authorship detection.

## Slide 2 Problem statement

Speaker: YASH RAJ AGARWAL. Approximate slot: 45 seconds including visual cues.

Cue: Keep eye contact; use the slide as a visual anchor.

The question we explore is whether measurable code style and structure can distinguish labeled examples. Code differs from ordinary prose: assignments specify algorithms, libraries introduce standard expressions, and programmers often produce similar valid solutions. A student can also add comments or rename identifiers without changing the program. These facts make the relationship between style and origin uncertain. Academic integrity motivates the question, but a probability alone cannot determine whether someone violated a policy. Our scope is to study a transparent classification pipeline and examine its failure modes, rather than automate academic decisions.

## Slide 3 Motivation and objectives

Speaker: YASH RAJ AGARWAL. Approximate slot: 45 seconds including visual cues.

Cue: Keep eye contact; use the slide as a visual anchor.

We selected a compact design that we can explain from source code to prediction. Our first objective is to turn code into numerical features using static analysis. Our second is to compare Logistic Regression, a Decision Tree and a Random Forest under the same data protocol. Our third is to connect the selected saved pipeline to a website and test how its outputs respond to controlled changes. The resulting contribution is an educational system with reproducible experiments. A stronger authorship study would require verified data, independent task families and a more rigorous evaluation protocol, which we discuss later.

## Slide 4 Literature and research gap

Speaker: YASH RAJ AGARWAL. Approximate slot: 45 seconds including visual cues.

Cue: Keep eye contact; use the slide as a visual anchor.

Our literature record separates direct evidence from background. Reference one is a C# code-stylometry study. Reference two examines text detector reliability and therefore motivates caution rather than supplying code-detection accuracy. Reference three provides source-code authorship comparison context, while reference four is the Random Forest foundation. We did not transfer their numerical findings into our own results. The accessible review did not verify twenty to twenty-five qualifying indexed journal articles, and the report states that gap explicitly. The immediate gap for our project is also empirical: we need a provenance-verified, task-diverse dataset before claiming real authorship performance.

## Slide 5 System architecture

Speaker: YASH RAJ AGARWAL. Approximate slot: 45 seconds including visual cues.

Cue: Trace the editable architecture arrows.

Follow the diagram from left to right. The React client sends source code and the selected Python language as JSON. FastAPI validates the request length, language and syntax. The backend extracts static features and constructs the one-row input expected by the saved preprocessing pipeline. TF-IDF transforms the lexical part and StandardScaler transforms the numerical part. The selected classifier returns class probabilities. The backend finds the AI class by name and returns the estimate plus feature summaries. The source never executes, and no external language-model API is called. Training is a separate offline process, shown below the inference path.

## Slide 6 Dataset and partitions

Speaker: YASH RAJ AGARWAL. Approximate slot: 45 seconds including visual cues.

Cue: Point to the actual data or diagram while explaining.

This slide is essential for interpreting every score. The CSV contains two hundred synthetic records generated from a few code templates. Cleanup removes forty duplicate rows, leaving one hundred and sixty unique examples, eighty per label. Group-based splitting produces ninety-six training, thirty-two validation and thirty-two test examples. The groups are disjoint by identifier, but they are not twenty independent algorithmic tasks. Shared template families remain across partitions, and the labels describe designed style categories rather than actual human or LLM provenance. I will now hand over to Nishaanth to explain exactly how those code samples become machine learning inputs.

## Slide 7 Feature engineering

Speaker: NISHAANTH S. Approximate slot: 45 seconds including visual cues.

Cue: Keep eye contact; use the slide as a visual anchor.

The numeric extractor combines style, structure and size measurements. It counts identifiers by occurrence and computes their mean length and naming ratios. It measures line length, comments and blank lines. It uses Python AST nodes to count functions, loops, conditionals, classes, exceptions and docstrings, together with a selected-construct nesting measure. Finally, it records source length, token count and unique-token ratio. This is eighteen numerical features in total. Naming entropy and cyclomatic complexity are not in the implementation. We explain them as possible future features, but do not present them as measured outputs. TF-IDF adds separate lexical coordinates.

## Slide 8 Mathematical foundations

Speaker: NISHAANTH S. Approximate slot: 45 seconds including visual cues.

Cue: Keep eye contact; use the slide as a visual anchor.

The equations here match the code. Sublinear term frequency dampens repeated words, and smoothed inverse-document frequency increases the weight of less common terms. Each lexical vector is then L2-normalized. Numerical features use training means and standard deviations. Logistic Regression maps a weighted score through the sigmoid. For example, a score of zero gives probability one half, while a score of zero point six gives about zero point six four six. The tree uses Gini impurity by default, not an entropy feature. The inference threshold is one half. The confidence displayed by the website is an uncalibrated model output, not an authorship guarantee.

## Slide 9 Three implemented models

Speaker: NISHAANTH S. Approximate slot: 45 seconds including visual cues.

Cue: Keep eye contact; use the slide as a visual anchor.

Logistic Regression gives a compact weighted baseline over the combined transformed feature vector. The standalone Decision Tree forms feature-based branches, constrained to depth twelve and at least two samples per leaf. Random Forest averages one hundred and sixty randomized trees with a minimum leaf size of two. All three use balanced class weights and seed forty-two. Shared numeric scaling simplifies the pipeline, although trees do not require it. We did not implement XGBoost or CodeBERT. Those were proposed in the original synopsis, and should be described as possible future extensions if a faculty member asks about them.

## Slide 10 Training and model selection

Speaker: NISHAANTH S. Approximate slot: 45 seconds including visual cues.

Cue: Keep eye contact; use the slide as a visual anchor.

The first grouped split reserves test groups, and the second reserves validation groups from the remainder. Each candidate fits its own transformation and classifier on training examples only. Validation F1, with AI as the positive class, is the main selection criterion. Accuracy breaks an F1 tie, and insertion order resolves a remaining tie. On this fixture all three candidates tie, so Logistic Regression is selected because it is listed first. The saved model remains fitted on the training subset. There is no cross-validation, grid search or later train-plus-validation refit in production. The separate experiment runner compares configurations without overwriting the deployed model.

## Slide 11 Actual model comparison

Speaker: NISHAANTH S. Approximate slot: 45 seconds including visual cues.

Cue: Point to the actual data or diagram while explaining.

The table reports the real saved evaluation output. Every candidate scores one point zero on accuracy, precision, recall and F1 on the thirty-two test examples. The selected matrix contains sixteen correctly predicted human-style and sixteen correctly predicted AI-style records. There are no errors on this test fixture. That is a limited observation, not a universal performance claim. The generator deliberately repeats strong style cues such as docstrings, and related templates occur across group partitions. Therefore the apparent perfection reflects an easy artificial task. We cannot infer the error rate on actual student submissions or code from a new model.

## Slide 12 Explainability and its limits

Speaker: NISHAANTH S. Approximate slot: 45 seconds including visual cues.

Cue: Point to the actual data or diagram while explaining.

The global importance chart sorts absolute coefficients for the selected Logistic Regression model. Code length, docstring count and identifier length rank prominently in this fitted fixture. Absolute values lose direction, and correlated measurements can share predictive weight. The sample feature panel is even more limited: it describes the submitted code but is not a complete local explanation of the prediction. We have not implemented SHAP or permutation importance. Renaming can change several measurements simultaneously, so one probability change does not identify one causal feature. I will now hand over to Aman for the working application, controlled demonstrations and conclusions.

## Slide 13 Website demonstration

Speaker: AMAN SHARMA. Approximate slot: 45 seconds including visual cues.

Cue: Switch between the running views; use screenshots if needed.

The screenshots are from our running application. In the live demonstration I will paste valid Python and press Analyze. The request goes to FastAPI, which loads the saved pipeline and returns the prediction. I will point to the AI-class probability and the extracted structural and style signals. I will then open Model Performance to show the actual saved comparison table and confusion matrix, and About to show the scope and limitations. The interface has three views rather than separate landing and prediction systems. If the server is unavailable, these screenshots and recorded JSON outputs provide an honest demonstration fallback.

## Slide 14 Controlled equivalent-code edits

Speaker: AMAN SHARMA. Approximate slot: 45 seconds including visual cues.

Cue: Point to the actual data or diagram while explaining.

The baseline function filters even integers. We apply several changes while preserving that behavior for ordinary finite integer lists. Long camelCase names change identifier statistics, code length and lexical words. Adding comments changes comment ratio and lexical content. Compact spacing changes line and source length while much AST structure remains unchanged. Replacing the loop with a comprehension changes the selected AST counts. The exact before-and-after probabilities are recorded in the experiment CSV and shown here. Some changes alter the predicted class and others only alter probability. We did not force any label switch; unchanged decisions are meaningful observations too.

## Slide 15 Configuration experiments

Speaker: AMAN SHARMA. Approximate slot: 45 seconds including visual cues.

Cue: Point to the actual data or diagram while explaining.

We also change the learning configuration reproducibly. Stronger and weaker regularization change the baseline probability while all tested validation class metrics remain perfect. Ablation retrains after removing defined feature groups. Those groups overlap conceptually with remaining length and token statistics, so the results are conditional comparisons rather than clean isolation of every notion of style. The tested thresholds leave validation precision and recall unchanged because no validation scores cross the boundaries. Finally, a factorial function probes a different task. We can report its output, but because it has no verified authorship label, we cannot turn that single probe into a generalization accuracy.

## Slide 16 Limitations and future scope

Speaker: AMAN SHARMA. Approximate slot: 45 seconds including visual cues.

Cue: Keep eye contact; use the slide as a visual anchor.

The strongest limitation is the dataset, not the choice of classifier. Synthetic labels and repeated templates make the measured problem much easier than real authorship detection. Probability outputs are uncalibrated. Equivalent refactoring can change estimates, and there is no evidence of performance on unseen generators, languages or real assignments. Our first improvement should be consented human submissions and matched generated outputs with known provenance. We should split by meaningful task families and authors where appropriate, repeat grouped evaluation and keep an independent final test set. Only then should we compare advanced models, richer AST features, entropy or extra languages.

## Slide 17 Conclusion

Speaker: AMAN SHARMA. Approximate slot: 45 seconds including visual cues.

Cue: Keep eye contact; use the slide as a visual anchor.

We have completed the computational workflow from source input through static features and fitted transformations to saved-model inference and a working website. We have also made the results and sensitivity experiments reproducible. The experiments show both stable and changing predictions under controlled modifications, which helps explain the relationship between features and learned behavior. The actual claim we can defend is an educational prototype with transparent limitations. We cannot defend operational authorship reliability from this dataset. That distinction is central to the project, because a clear explanation of what the experiment establishes is more valuable than an impressive-looking accuracy number.

## Slide 18 References and questions

Speaker: AMAN SHARMA. Approximate slot: 45 seconds including visual cues.

Cue: Keep eye contact; use the slide as a visual anchor.

Thank you. We are ready to discuss the mathematics, source code, data protocol and reliability concerns. Yash can explain the motivation, literature and dataset. Nishaanth can walk through tokenization, AST measurements, TF-IDF, scaling and the classifiers. I can demonstrate the API, saved inference, evaluation and experiments. If a question asks why we chose XGBoost, the accurate answer is that it is not implemented. If asked why the scores are perfect, we will explain the repeated synthetic templates. The references and reproduction scripts are available in the evaluation package for inspection.

