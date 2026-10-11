# NHANES Undiagnosed Diabetes 2021–2023

Data, analysis scripts and saved results for the Group 1 project on identifying undiagnosed diabetes in U.S. adults using the NHANES August 2021–August 2023 cycle.

## Repository contents

| Location | Contents |
| --- | --- |
| `data/raw/` | 24 NHANES public-use SAS transport files (`.XPT`). |
| `data/nhanes_diabetes_2021_2023.csv` | Merged participant-level table with 6,001 rows and 88 columns. |
| `scripts/nhanes_hba1c_pycharm.py` | Python analysis comparing logistic regression, random forest and XGBoost with four class-imbalance approaches using nested cross-validation. |
| `scripts/aim3_survey_analysis.R` | R analysis of survey-weighted prevalence and associations by age, sex, health insurance and income. |
| `results/aim1/` | Model comparisons, fold-level metrics, out-of-fold predictions and run metadata. |
| `results/aim3/` | Prevalence estimates, Rao–Scott tests, regression estimates and supplementary analysis tables. |
| `requirements.txt` | Python package versions recorded with the saved Aim 1 results. |

## Data source

The source files are from the National Health and Nutrition Examination Survey (NHANES), conducted by the U.S. National Center for Health Statistics, Centers for Disease Control and Prevention.

- [NHANES August 2021–August 2023 data and documentation](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023)
- [Glycohemoglobin documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.htm)
- [Diabetes questionnaire documentation](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DIQ_L.htm)

`SEQN` is the participant identifier in the source files and corresponds to `participant_id` in the merged table. Each source module has its own CDC codebook.

## Analysis setup

### Python

The Python script uses Python 3.10 or later. Install the dependencies with:

```sh
python -m pip install -r requirements.txt
```

Configure `DATA_FILE` at the top of `scripts/nhanes_hba1c_pycharm.py` to select the analysis input, and configure `OUTPUT_FOLDER` for new results. Absolute paths are supported; relative paths are resolved from the script directory. The script validates required columns and cohort rules before fitting models.

```sh
python scripts/nhanes_hba1c_pycharm.py
```

### R

The survey analysis uses the R package `survey`. The script installs it from CRAN if it is not already available. An analysis input can be supplied explicitly:

```sh
Rscript scripts/aim3_survey_analysis.R path/to/analysis_input.csv
```

In this directory structure, the R script writes new outputs to `reports/aim3/`. The files under `results/` contain the saved project outputs.

## Saved results

Aim 1 includes `comparison.csv`, `fold_metrics.csv`, `nested_selection_metrics.csv`, `oof_predictions.csv` and `data_audit.json`.

Aim 3 includes `Table_Aim3.csv`, `prevalence_by_group.csv`, `rao_scott_tests.csv`, `regression_models.csv` and `supplementary_among_diabetes.csv`.

## Project

Group 1 — BMI5109 Advanced Scientific Research Methods, MSc in Biomedical Informatics, National University of Singapore.
