# Predictive Modelling for Lung Cancer Treatment Selection Based on Molecular Profile
## Project Overview 
Personalised medicine is transforming healthcare by tailoring treatments to individual patients. In modern precision oncology, for instance, clinicians use a patient’s molecular profile (e.g., genomics, transcriptomics, proteomics) to guide the choice of targeted therapy.

The project has three primary objectives:
- **Predictive Potential of Molecular Profiles:** Determine if and how well molecular profiling data can be used to predict a patient’s treatment type or the optimal therapy choice. In other words, investigate whether patterns in molecular markers correlate strongly enough with treatment outcomes to enable accurate predictions.
- **Comparison of Different Classifiers:** Apply a range of machine learning classification models to the prediction task and evaluate which algorithms perform most effectively. By trying multiple model types, we can identify if certain approaches (e.g., tree-based vs. linear models) are better suited for this biomedical prediction problem.
- **Individual Models vs. Ensemble Approach:** Compare the performance of individually optimised classifiers (each tuned via hyperparameter optimisation) against an ensemble method. The ensemble method, termed `HyperParamEnsemble`, involves training multiple instances of the same type of model with different hyperparameter settings and then combining their predictions by voting. This tests whether aggregating several tuned models of one kind can outperform a single best-tuned model of that kind.

If successful, such a predictive model could become a clinical decision-support tool, helping to verify or even challenge standard treatment recommendations for individual patients. This would mean that, given two or more possible treatments, an algorithm could suggest which one is likely to work best for a specific patient based on their unique biomarker signature.

## Repository Structure

`Predictor_of_LUAD_treatments.ipynb` — Main notebook implementing the full experimental pipeline for treatment classification.

`MSK_CHORD.csv` — Processed LUAD dataset snapshot produced by the notebook (patient-level table integrating selected clinical variables, binary indicators of recurrent genomic alterations, and treatment category labels), then reloaded for downstream preprocessing and modelling.

`project_presentation.pdf` — Short slide deck summarising the motivation, setup, modelling approach, and key findings.

`README.md` — Project overview, methodology, and experimental results.


## Data Description
**Dataset:** [MSK-CHORD (Nature 2024)](https://www.cbioportal.org/study/summary?id=msk_chord_2024), a large multi-institutional oncology cohort accessible via cBioPortal. It comprises clinical, genomic and treatment data for around 25,000 cancer patients. This analysis focuses on the LUAD (lung adenocarcinoma) subcohort.

**Data Files Used:** `data_clinical_patient.txt`, `data_clinical_sample.txt`, `data_mutations.txt`, `data_timeline_treatment.txt`

**Cohort Filter:**
  - Full LUAD cohort: 4,463 patients with treatment annotations.
  - Single-therapy subset: 1,300 patients receiving exactly one therapy category.

**Features:**
  - Genomic: Binary mutation indicators for 23 genes mutated in ≥5% of LUAD samples (e.g., TP53, EGFR, STK11).
  - Clinical: Patient age, tumour stage, smoking status, etc. (one-hot encoded)

**Outcome Label:** Treatment category per patient (either a single category in the 5-class subset or a combination label in the 11-class setting). Rare combinations were grouped into “Other” or excluded when constructing the single-therapy subset.

**Additional dataset (sanity check):** **scikit-learn Digits** (n=1797, 10-class handwritten digits) to validate the modelling pipeline *(expected high accuracy)*, helping isolate dataset difficulty from implementation issues.

> [!NOTE]
> *Interpretation note:* The substantial performance gap between the 11-class multi-therapy setting and the 5-class single-therapy subset supports the hypothesis that molecular profiles have predictive potential for treatment type. In the single-therapy subset, each patient’s profile is linked to one dominant treatment modality, yielding a cleaner mapping between features and label. In contrast, multi-therapy labels effectively mix multiple treatment signals within a single patient profile, making the learning problem more ambiguous and weakly supervised.

## Project Workflow
1. **Data Ingestion & Preprocessing** 
- Loading clinical patient data, sample metadata, mutation calls, and treatment timelines from the MSK-CHORD files.
- Filtering to lung adenocarcinoma patients and aligning records by patient ID.
- Merging all data sources into a single **patient-level table**.

2. **Label Engineering (Therapy Categories)**
- Define five therapy subtypes: **Chemo, Immuno, Molecular, Supportive, Investigational**.
- Transform each patient’s timeline into a label: sort events by date per patient, collect unique therapy subtypes, and join them with '+' if multiple were received (e.g., Chemo+Molecular for a patient who had both).
- Class simplification: merge similar subtypes into a common taxonomy (e.g., targeted/biologic → Molecular).
- Normalise combination labels to a consistent order (to avoid duplicate classes like Immuno+Chemo vs Chemo+Immuno).
- Build two target settings:
  - **Multi-treatment labels (11 classes):** Top 10 most frequent therapy combinations + an “Other” category for all rarer combinations.
  - **Single-treatment labels (5 classes):** Keep only patients with exactly one therapy subtype. 

3. **Feature Engineering**
- **Genomic features:** Pivot the mutation list into a binary matrix of patients × genes. Keep genes mutated in >5% of LUAD patients to ensure relevance (yielding 23 genomic features).
- **Clinical features:** Select informative clinical variables (e.g., age, stage, smoking history). Encode categorical variables as one-hot vectors and scale continuous variables. Remove any columns that leak the target or are irrelevant for prediction.

  Concatenate genomic and clinical features for each patient to form the final feature vector.

4. **Setup datasets** Build training/evaluation datasets for different scenarios:
  - `msk_chord`: LUAD multi-treatment (top-10 combos + Other → 11 classes).
  - `msk_chord_filtered`: LUAD single-treatment only (5 classes).
  - `digits`: scikit-learn Digits (10 classes) used as a sanity-check benchmark.

5. **Model Training** Two modelling pipelines are executed in parallel to compare performance:
- **Approach A — Individual classifiers (6 models)** Train and tune a variety of standalone classification algorithms:
  - **Models explored:**
    - Decision Tree
    - Random Forest
    - Support Vector Machine (SVM)
    - Bernoulli Naive Bayes
    - Logistic Regression
    - CatBoost (gradient boosting)

   - **Hyperparameter Tuning:** Each model is optimised with grid search (GridSearchCV, 3-fold cross-validation on the training data) to find the best hyperparameters.
  - **Model Evaluation (CV):** After tuning, each model’s best estimator is re-evaluated using an additional cross-validation (3-fold) on the respective dataset. Performance metrics recorded include Accuracy and Macro F1-score (to account for class imbalance).
    
- **Approach B — HyperParamEnsemble (6 models)** Train an ensemble of multiple classifier instances (homogeneous ensemble) for each algorithm type:
  - Custom scikit-learn-compatible estimator `HyperParamEnsembleClassifier`, which builds an ensemble from one base algorithm (e.g., an ensemble of SVMs, an ensemble of RFs, etc.).
  - **Ensemble Construction:** For a given base model type, train several instances across a grid of hyperparameters *(covering the same search space as in Approach A)*. The ensemble combines these members’ predictions by **voting**:
      - *Hard*: each member votes for a class label, and the majority class is the ensemble prediction.
      - *Soft*: average the predicted class probabilities from all members and pick the argmax as the final prediction.
   
   - **Model Evaluation (Hold-out):** Each `HyperParamEnsemble` is evaluated on a single stratified train/test split (80/20). Performance is measured with Accuracy and Macro F1-score on the held-out test set (no cross-validation, since the ensemble itself spans multiple models).

6. **Evaluation & Comparison**
  - Aggregate all metrics (accuracy, F1, etc.) per model and dataset into comparison tables for clarity.
  - **Visualise** model performance:
    - Heatmaps of Accuracy and Macro-F1 for each model vs. each dataset (facilitates quick performance scanning).
    - Bar charts summarising mean performance metrics by model type (to compare overall effectiveness).
 
  - **Statistical Analysis:** Conduct non-parametric tests to check for significant performance differences:
    - Perform a **Friedman test** across all classifiers (to detect overall differences in rankings on multiple datasets).
    - If significant, follow up with a **Nemenyi post-hoc** test to identify pairwise differences. This is visualised with a **Critical Difference (CD) diagram** (using the `aeon` toolkit), showing groups of models that are not significantly different in performance.

## Results Summary
For each dataset, the highest achieved Accuracy and Macro-F1 (with the model that achieved them):
| Dataset                                   | Classes |                   Best Accuracy                 |             Best Macro-F1           |
|-------------------------------------------|:-------:|:-----------------------------------------------:|:-----------------------------------:|
| Full MSK-CHORD (combo-therapy patients)   |   11    | Random Forest ~0.28                             | HyperParamEnsemble (CatBoost) ~0.18 |
| Filtered MSK-CHORD (single-therapy only)  |    5    | HyperParamEnsemble (DecisionTree and SVM) ~0.65 | HyperParamEnsemble (LogReg) ~0.44   |
| Digits (sanity check)                     |   10    | SVM ~0.99                                       | SVM ~0.99                           |


Top-3 Overall (mean across datasets):
| Metric (mean across datasets) | Rank |                Model                |
|-------------------------------|-----:|-------------------------------------|
| Accuracy                      | 1    | SVM and Random Forest               |
|                               | 2    | HyperParamEnsemble (SVM)            |
|                               | 3    | HyperParamEnsemble (Random Forest)  |
| Macro F1                      | 1    | HyperParamEnsemble (CatBoost)       |
|                               | 2    | HyperParamEnsemble (LogReg)         |
|                               | 3    | CatBoost                            |

**Key findings:**
- Single-therapy classification using molecular and clinical features was only moderately successful (highest ~65% accuracy and ~0.44 macro-F1), indicating limited predictive power for treatment selection from biomarkers.
- Macro F1 scores were substantially lower than accuracy on the LUAD datasets, reflecting class imbalance and uneven classifier performance across therapy types. 
- Single-therapy MSK-CHORD (sparse binary genomics + one-hot clinical): Accuracy was maximized by margin-/tree-based models (HyperParamEnsemble SVM/DecisionTree = 0.654, single SVM 0.651, RF 0.650), while class-balanced performance favored regularized/probabilistic models (HyperParamEnsemble LogReg macro-F1 = 0.442, CatBoost ensemble = 0.428); this pattern is consistent with sparse binary features + class imbalance (SVM/RF fit dominant decision boundaries → higher accuracy, LogReg/boosting generalize more evenly across classes → higher macro-F1), so prioritize SVM/RF for overall hit-rate and class-weighted LogReg/GBDT (CatBoost/XGBoost/LightGBM) when macro-F1 and minority-class recall matter.
- The single-therapy vs. multi-therapy gap is a signal check: when labels are “clean” (single modality), performance is substantially higher; when labels mix modalities, performance drops sharply due to label ambiguity. This supports the existence of predictive signal in molecular profiles for treatment modality.
- Sanity check (Digits dataset): models reach ~97–99% accuracy, indicating the implementation is correct and LUAD results are driven by task difficulty.
- No model had a statistically significant edge overall – a Friedman test on all results yielded p ≈ 1.0 – suggesting that performance differences among classifiers were not meaningful across datasets.
- Individually tuned vs. HyperParamEnsemble (Digits + single-therapy): ensembles did not consistently outperform the best single tuned model; the effect was metric- and model-dependent. On the filtered LUAD cohort, gains were split (≈half improved, half degraded), while on Digits the ensemble more frequently improved both accuracy and macro-F1. Overall, the ensemble behaves as a robustness trade-off rather than a guaranteed boost in top-line accuracy.

> [!NOTE]
> *Discussion (Hyperparameter Search)* In theory, ensembling should provide better generalization and performance than individual models, as it combines multiple models trained with different hyperparameters, which helps reduce variance and bias. However, ensembling did not provide significant improvement. One of the main reasons was the hyperparameter selection strategy. Used `GridSearchCV` for individual classifiers, which exhaustively searches through all possible hyperparameter combinations to find the optimal configuration. At the same time, `HyperParamEnsembleClassifier` does not perform a full search but instead randomly selects a limited number of hyperparameter combinations (equal to the number of models in the ensemble) and averages the predictions through voting. This created a slight bias in favor of individual models, as they benefited from a more detailed hyperparameter search, whereas the ensemble depended on the random selection of configurations. A fairer comparison would have been to use `RandomizedSearchCV` for individual models.

> [!NOTE]
> *Discussion (Compute & runtime)* Model training was computationally demanding, since fitting and tuning required substantial resources. Runtime was highly model- and hyperparameter-dependent; cross-validation was kept fixed (as a deliberate trade-off between reliability and cost), yet exhaustive tuning still dominated total compute. However, `HyperParamEnsembleClassifier` demonstrated decent efficiency — it maintained fairly high results with lower computational costs (presumably due to a small number of estimators) compared to tuning hyperparameter combinations for an individual models (the results were slightly better, but training took significantly longer). The ensemble also outperformed some individual classifiers on datasets, which indicates its ability to find diverse hyperparameter configurations.

## Future Work
- **Scale up and refine single-therapy data (highest impact for hypothesis testing)**
  - Collect **larger single-therapy cohorts** (more patients per class).
  - Move from therapy categories to **drug-/regimen-level labels** (specific agents, lines of therapy, dose/schedule where possible) to reduce label noise and strengthen the genotype–treatment signal. 

- **Switch to outcome-driven targets (from “what was prescribed” to “what works”)**
  - Train models on **response vs. resistance** (e.g., RECIST/ORR/DCR) rather than historical treatment choice.
  - Formulate per-treatment tasks: *“Is patient likely to benefit from therapy X?”* (more clinically meaningful than predicting physician decisions). 

- **Add survival analysis (mandatory extension)**
  - Model **time-to-event endpoints** (OS/PFS) with survival methods (e.g., CoxPH / Random Survival Forest / DeepSurv).
  - Evaluate with **C-index** and time-dependent AUC; enable treatment-effect comparisons via survival stratification. 

- **Improve representation of sparse molecular features**
  - Apply **feature selection** (frequency/informativeness) + **regularisation** (L1/ElasticNet).
  - Engineer **pathway-level features** (aggregate mutations into functional pathways) to reduce sparsity and increase biological signal.

- **Smarter feature engineering (biology + clinic)**
  - Add composite biomarkers (e.g., **TMB/MSI**, driver counts, actionable alterations if available).
  - Model interactions **(clinical × molecular)** and improve preprocessing (missingness handling, leakage control, calibration).
    
- **Handle class imbalance explicitly**
  - Use **class weights**, resampling (over/under-sampling), and threshold tuning.
  - Prioritise **macro-F1 / per-class recall / PR-AUC** to avoid majority-class dominance. 

- **Richer hyperparameter optimisation + stricter evaluation**
  - Replace coarse grids with **Bayesian optimisation** (e.g., Optuna) and widen the search space.
  - Use **nested CV** (or robust external holdout) to prevent optimistic tuning bias.

- **Add richer omics modalities (as separate modelling tracks)**
  - Explore pipelines using **gene expression** (transcriptomics) and **proteomics** (or in parallel).
  - Compare: mutation-only vs. expression-only vs. multi-omics fusion (late/early fusion).

- **Dimensionality reduction for stability and generalisation**
  - Apply **PCA** (and/or other DR methods) on high-dimensional omics features to reduce noise and improve conditioning.
  - Validate DR choices via CV and assess impact on macro-F1 and calibration.

- **Broader, disease-specific cohorts + external validation**
  - Expand LUAD cohorts across centers/time and validate on **independent datasets** to test transportability.
  - Stratify by clinically meaningful subgroups (stage, oncogenic drivers, line of therapy).

## Environment & Dependencies
This project is implemented in Python (Jupyter/Colab) and uses a standard ML stack for tabular modelling, evaluation, and visualisation.

**Core scientific stack:**
- `pandas` — tabular data loading/cleaning/merging (patient-level tables, feature matrices)
- `numpy` — numerical computing backbone (arrays, vectorised ops)

**Visualisation & reporting:**
- `matplotlib`, `seaborn` — plots and statistical graphics (e.g., heatmaps, metric summaries)
- `tabulate` — readable result tables in notebook/console
- `IPython.display` — rich notebook rendering (optional; notebook-only)

**Modelling, preprocessing & evaluation:**
- `scikit-learn` (`sklearn`) — core ML framework used for:
  - Estimators: `LogisticRegression`, `BernoulliNB`, `DecisionTreeClassifier`, `RandomForestClassifier`, `SVC`
  - Model selection & evaluation: `train_test_split`, `cross_validate`, `cross_val_score`, `GridSearchCV`, `ParameterGrid`
  - Preprocessing: `SimpleImputer`, `MinMaxScaler`, `LabelEncoder`
  - Metrics & reports: `accuracy_score`, `f1_score`, `classification_report` (plus `sklearn.metrics` utilities)
  - Custom estimator support: `BaseEstimator`, `ClassifierMixin`, and validation helpers (`check_X_y`, `check_array`, `check_is_fitted`)
  - Utilities / datasets: `Bunch` and toy datasets for sanity checks (e.g., `load_digits`)

- `catboost` — gradient-boosted decision trees baseline (`CatBoostClassifier`)

**Statistical comparison:** `aeon` — non-parametric statistical tests and visualisations for model comparison (Friedman/Nemenyi workflow, critical difference diagrams)
  
**Standard library utilities:** `random`, `itertools`, `time`, `pickle`, `inspect`, `tarfile`, `pathlib`

### Installation (pip)
Most dependencies are standard in common Python data-science environments.
The notebook explicitly installs two additional packages: `aeon`, `catboost`.

```bash
pip install -U numpy pandas scikit-learn matplotlib seaborn tabulate catboost aeon
```

