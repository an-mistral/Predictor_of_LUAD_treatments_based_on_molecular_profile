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

**Cohort Filter:**
  - Full LUAD cohort: 4,463 patients with treatment annotations (multi-therapy combinations allowed → 11 classes).
  - Single-therapy subset: 1,300 patients receiving exactly one therapy category → 5 classes (Chemotherapy, Immunotherapy, Molecular/Targeted, Supportive, Investigational).

**Features:**
  - Genomic: Binary mutation indicators for 23 genes mutated in ≥5% of LUAD samples (e.g., TP53, EGFR, STK11).
  - Clinical: Patient age, tumour stage, smoking status, etc. (one-hot encoded)

**Outcome Label:** Treatment category per patient (either a single category in the 5-class subset or a combination label in the 11-class setting). Rare combinations were grouped into “Other” or excluded when constructing the single-therapy subset.

**Additional dataset (sanity check):** scikit-learn Digits (10 classes) to validate the modelling pipeline (expected high accuracy), helping isolate dataset difficulty from implementation issues.

> [!NOTE]
> *Interpretation note:* The substantial performance gap between the 11-class multi-therapy setting and the 5-class single-therapy subset supports the hypothesis that molecular profiles have predictive potential for treatment type. In the single-therapy subset, each patient’s profile is linked to one dominant treatment modality, yielding a cleaner mapping between features and label. In contrast, multi-therapy labels effectively mix multiple treatment signals within a single patient profile, making the learning problem more ambiguous and weakly supervised.

## Project Workflow

## Results Summary

## Requirements


