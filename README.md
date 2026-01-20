# Predictive Modelling for Lung Cancer Treatment Selection Based on Molecular Profile
## Project Overview 
Personalised medicine is transforming healthcare by tailoring treatments to individual patients. In modern precision oncology, for instance, clinicians use a patient’s molecular profile (genomic and proteomic markers) to guide the choice of targeted therapy.

The project has three primary objectives:
- **Predictive Potential of Molecular Profiles:** Determine if and how well molecular profiling data can be used to predict a patient’s treatment type or the optimal therapy choice. In other words, investigate whether patterns in molecular markers correlate strongly enough with treatment outcomes to enable accurate predictions.
- **Comparison of Different Classifiers:** Apply a range of machine learning classification models to the prediction task and evaluate which algorithms perform most effectively. By trying multiple model types, we can identify if certain approaches (e.g., tree-based vs. linear models) are better suited for this biomedical prediction problem.
- **Individual Models vs. Ensemble Approach:** Compare the performance of individually optimised classifiers (each tuned via hyperparameter optimisation) against an ensemble method. The ensemble method, termed `HyperParamEnsemble`, involves training multiple instances of the same type of model with different hyperparameter settings and then combining their predictions by voting. This tests whether aggregating several tuned models of one kind can outperform a single best-tuned model of that kind.

If successful, such a predictive model could become a clinical decision-support tool, helping to verify or even challenge standard treatment recommendations for individual patients. This would mean that, given two or more possible treatments, an algorithm could suggest which one is likely to work best for a specific patient based on their unique biomarker signature.

## Repository Structure
```text
Predictor_of_LUAD_treatments_based_on_molecular_profile/
├── Predictor_of_LUAD_treatments.ipynb
├── MSK_CHORD.csv
├── project_presentation.pdf
└── README.md
```
`Predictor_of_LUAD_treatments.ipynb` — Main notebook implementing the full experimental pipeline for treatment classification.

`MSK_CHORD.csv` — Processed LUAD dataset snapshot produced by the notebook (patient-level table integrating selected clinical variables, binary indicators of recurrent genomic alterations, and treatment category labels), then reloaded for downstream preprocessing and modelling.

`project_presentation.pdf` — Short slide deck summarising the motivation, setup, modelling approach, and key findings.

`README.md` — Project overview, methodology, and experimental results.


## Data Description
**Dataset:** [MSK-CHORD (Nature 2024)](https://www.cbioportal.org/study/summary?id=msk_chord_2024), a large multi-institutional oncology cohort accessible via cBioPortal. It comprises clinical, genomic and treatment data for around 25,000 cancer patients. This analysis focuses on the LUAD (lung adenocarcinoma) subcohort.

**Files used:** `data_clinical_patient.txt`, `data_clinical_sample.txt`, `data_mutations.txt`, `data_timeline_treatment.txt`

**Cohort Filter:**
  - Full LUAD cohort: 4,463 patients with treatment annotations (multi-therapy combinations allowed → 11 classes).
  - Single-therapy subset: 1,300 patients receiving exactly one therapy category → 5 classes.

**Features:**
  - Genomic: Binary mutation indicators for 23 genes mutated in ≥5% of LUAD samples (e.g., TP53, EGFR, STK11).
  - Clinical: Patient age, tumour stage, smoking status, etc. (one-hot encoded)

**Outcome Label:** Treatment category per patient (either a single category in the 5-class subset or a combination label in the 11-class setting). Rare combinations were grouped into “Other” or excluded when constructing the single-therapy subset.

**Additional dataset (sanity check):** scikit-learn Digits (10 classes) to validate the modelling pipeline (expected high accuracy), helping isolate dataset difficulty from implementation issues.

> [!NOTE]
> *Interpretation note:* The substantial performance gap between the 11-class multi-therapy setting and the 5-class single-therapy subset supports the hypothesis that molecular profiles have predictive potential for treatment type. In the single-therapy subset, each patient’s profile is linked to one dominant treatment modality, yielding a cleaner mapping between features and label. In contrast, multi-therapy labels effectively mix multiple treatment signals within a single patient profile, making the learning problem more ambiguous and weakly supervised.

## Project Workflow
1. **Data acquisition and cohort construction**
   - Load MSK-CHORD study tables from an extracted archive.
   - Restrict the cohort to LUAD samples.
   - Align modalities at the patient level.

2. **Label engineering: treatment categories**
   - Define therapy subtypes: **Chemo**, **Immuno**, **Molecular**, **Supportive**, **Investigational**.
   - Derive patient-level treatment labels from the treatment timeline: sort per-patient events by date and aggregate unique therapy subtupe values into a single label using '+' concatenation (chronology-aware, de-duplicated).
   - Class simplification: merge similar subtypes into a common taxonomy (e.g., targeted/biologic → Molecular; hormone/bone-strengthening → Supportive).
   - Normalize combination labels to a canonical ordering to avoid equivalent labels with different orderings.
    - Build two target settings:
    - **Multi-treatment label setting**: keep the **top-10 most frequent combinations**, collapse the rest into **“Other”** → **11 classes** total 
    - **Single-treatment setting**: keep only patients with exactly one of {Chemo, Immuno, Molecular, Supportive, Investigational} → **5 classes** 

3. **Feature engineering**
  - **Genomic features**
      - Pivot mutations into a binary patient × gene matrix (0/1 indicator per gene).
      - Filter genes by prevalence: keep genes mutated in >5% of LUAD patients.
  - **Clinical features**
      - Create binary clinical indicators
      - Scale numeric clinical variables
      - Drop leakage / non-feature columns

4. **Setup datasets**
  - `msk_chord`: LUAD multi-treatment (top-10 combos + Other → 11 classes)
  - `msk_chord_filtered`: LUAD single-treatment only (5 classes)
  - `digits`: scikit-learn Digits (10 classes) used as a sanity-check benchmark

5. **Two modelling approaches (12 total models)**
  - **Approach A — Individual classifiers**
    - Decision Tree
    - Random Forest
    - SVM, BernoulliNB
    - Logistic Regression
    - CatBoost
    
    Each model is tuned with **GridSearchCV** (CV=3), then re-evaluated with a **final cross-validation** (CV=3) using **Accuracy** + **Macro-F1**
    
  - **Approach B — HyperParamEnsemble (6 models)**
    - Custom scikit-learn-compatible estimator `HyperParamEnsembleClassifier`
    - Trains multiple instances of the same base algorithm across a parameter grid and aggregates predictions via voting:
      - **Hard voting**: majority class
      - **Soft voting**: mean predicted probabilities then argmax
    - Evaluated via a **single stratified train/test split** (80/20), reporting **Accuracy** + **Macro-F1**

6. **Evaluation, comparison, and reporting**
  - Collect per-(dataset, model) metrics into structured tables.
  - Visualize model performance:
    - Heatmaps for Accuracy and Macro-F1 across datasets/models
    - Bar charts of mean metrics across datasets
  - Run statistical comparisons across models using **Friedman test** and **Nemenyi post-hoc** (via `aeon`) and plot a **Critical Difference diagram**.


## Results Summary

## Requirements


