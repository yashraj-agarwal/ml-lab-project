# Technical Inventory

200 synthetic CSV rows; 40 duplicate rows removed; 160 unique examples (80 human-style, 80 AI-style); 20 synthetic problem groups; 96 training, 32 validation, 32 test samples.

Both labels are synthetic style categories written into a generator, not verified human or LLM provenance. The 20 group IDs vary a few filtering templates and do not represent 20 independent algorithmic tasks. Perfect scores describe template separation only.

## Implemented

Static Python tokenization, regex identifiers, 18 numeric features, TF-IDF unigrams/bigrams, StandardScaler, Logistic Regression, Decision Tree, Random Forest, grouped holdout evaluation, saved selected inference pipeline, FastAPI endpoints and React views. Schema-2 model metadata records library versions and dataset hash.

## Not implemented

Automatic language identification; consented classroom dataset; verified human/LLM provenance; naming entropy; cyclomatic complexity; XGBoost; CodeBERT; supervised feature selector; production cross-validation/grid search; calibration; SHAP; permutation importance; cloud deployment. Experiment-only ROC AUC is distinct from production metrics.

## Evaluation caveats

No exact problem_id overlap across partitions, but shared template-family overlap remains. GroupShuffleSplit is not stratified. The single factorial probe is unlabeled. The selected classifier is fitted only on training data. All candidates tie on validation and test metrics; deterministic insertion order selects Logistic Regression. Only that model is saved.

## Corrections made

The unusual mixed comment-count/character expression was replaced by comment-token/code-line ratio. Invalid Python now yields 422. Contradictory duplicate labels are rejected. Each split must include both labels. Model schema/version metadata is stored. Frontend validation-error arrays become readable text. Tailwind was migrated to version 4 and Vite updated; the final npm audit reports no vulnerabilities. The static LOCAL MODEL badge remains branding, not a live health check.

## Artifact evidence

```json
{
  "python": "3.12.7",
  "sklearn": "1.9.1",
  "joblib": "1.6.0",
  "model_sha256": "a2e95d73c43426a9b76d821e03630aedd20007d5b0b95faa261526f548c23b79",
  "dataset": {
    "source_rows": 200,
    "removed_rows": 40,
    "unique_rows": 160,
    "class_counts": {
      "ai": 80,
      "human": 80
    },
    "group_count": 20
  },
  "selected_estimator_parameters": {
    "C": 1.0,
    "class_weight": "balanced",
    "dual": false,
    "fit_intercept": true,
    "intercept_scaling": 1,
    "l1_ratio": 0.0,
    "max_iter": 1500,
    "n_jobs": null,
    "penalty": "deprecated",
    "random_state": 42,
    "solver": "lbfgs",
    "tol": 0.0001,
    "verbose": 0,
    "warm_start": false
  },
  "tfidf_parameters": {
    "analyzer": "word",
    "binary": "False",
    "decode_error": "strict",
    "dtype": "<class 'numpy.float64'>",
    "encoding": "utf-8",
    "input": "content",
    "lowercase": "True",
    "max_df": "1.0",
    "max_features": "1200",
    "min_df": "1",
    "ngram_range": "(1, 2)",
    "norm": "l2",
    "preprocessor": "None",
    "smooth_idf": "True",
    "stop_words": "None",
    "strip_accents": "None",
    "sublinear_tf": "True",
    "token_pattern": "(?u)\\b\\w+\\b",
    "tokenizer": "None",
    "use_idf": "True",
    "vocabulary": "None"
  },
  "lexical_vocabulary_size": 362,
  "artifact_metadata": {
    "schema_version": 2,
    "python": "3.12.7",
    "scikit_learn": "1.9.1",
    "joblib": "1.6.0",
    "dataset_sha256": "2c02b806bf694a5ab8f4cf99a051dc3e5275b0f182654e19f88700401b11543d"
  },
  "split_groups": {
    "train": [
      "problem-03",
      "problem-04",
      "problem-05",
      "problem-06",
      "problem-07",
      "problem-08",
      "problem-10",
      "problem-12",
      "problem-13",
      "problem-14",
      "problem-17",
      "problem-19"
    ],
    "validation": [
      "problem-09",
      "problem-11",
      "problem-15",
      "problem-20"
    ],
    "test": [
      "problem-01",
      "problem-02",
      "problem-16",
      "problem-18"
    ]
  },
  "split_class_counts": {
    "train": {
      "ai": 48,
      "human": 48
    },
    "validation": {
      "human": 16,
      "ai": 16
    },
    "test": {
      "human": 16,
      "ai": 16
    }
  },
  "saved_metrics": {
    "artifact_metadata": {
      "schema_version": 2,
      "python": "3.12.7",
      "scikit_learn": "1.9.1",
      "joblib": "1.6.0",
      "dataset_sha256": "2c02b806bf694a5ab8f4cf99a051dc3e5275b0f182654e19f88700401b11543d"
    },
    "selected_model": "Logistic Regression",
    "selection_metric": "validation F1 (AI as positive class)",
    "validation_metrics": {
      "Logistic Regression": {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      },
      "Decision Tree": {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      },
      "Random Forest": {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      }
    },
    "models": {
      "Logistic Regression": {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      },
      "Decision Tree": {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      },
      "Random Forest": {
        "accuracy": 1.0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0
      }
    },
    "test_metrics": {
      "accuracy": 1.0,
      "precision": 1.0,
      "recall": 1.0,
      "f1": 1.0
    },
    "confusion_matrix": [
      [
        16,
        0
      ],
      [
        0,
        16
      ]
    ],
    "confusion_matrix_labels": [
      "human",
      "ai"
    ],
    "feature_importance": [
      {
        "feature": "structure: code_length",
        "importance": 1.14729
      },
      {
        "feature": "structure: docstring_count",
        "importance": 1.03232
      },
      {
        "feature": "structure: average_identifier_length",
        "importance": 0.95554
      },
      {
        "feature": "structure: average_line_length",
        "importance": 0.80101
      },
      {
        "feature": "structure: ast_node_count",
        "importance": 0.44572
      },
      {
        "feature": "structure: token_count",
        "importance": 0.43145
      },
      {
        "feature": "structure: unique_token_ratio",
        "importance": 0.38912
      },
      {
        "feature": "structure: loop_count",
        "importance": 0.36978
      },
      {
        "feature": "structure: conditional_count",
        "importance": 0.36978
      },
      {
        "feature": "structure: max_nesting_depth",
        "importance": 0.36978
      },
      {
        "feature": "structure: line_count",
        "importance": 0.32993
      },
      {
        "feature": "structure: comment_to_code_ratio",
        "importance": 0.24528
      }
    ],
    "dataset": {
      "source_rows": 200,
      "rows": 160,
      "removed_missing_invalid_blank_or_duplicate_rows": 40,
      "after_validation_and_duplicate_removal": 160,
      "class_distribution": {
        "ai": 80,
        "human": 80
      },
      "problem_groups": 20,
      "split_sizes": {
        "train": 96,
        "validation": 32,
        "test": 32
      },
      "source": "samples.csv"
    },
    "limitations": "Demonstration data is synthetic and metrics do not indicate real-world detection performance."
  }
}
```
