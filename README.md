# NHANES Undiagnosed Diabetes 2021–2023

Data, analysis code and saved results for the Group 1 study,
*Identifying Undiagnosed Diabetes in U.S. Adults from Routinely Collected Data:
A Machine Learning Study Using NHANES 2021–2023*.

## Current analyses

- **Aim 1:** compare logistic regression (LR), random forest (RF) and XGBoost
  using nine prespecified non-glycaemic predictors (ten encoded columns).
- **Aim 2:** describe survey-weighted undiagnosed diabetes prevalence and
  adjusted associations by age, sex, health insurance and income.

The former Aim 2 was removed; the former Aim 3 is now Aim 2. The two current
analyses use the same 5,026 included participant IDs, including 126 positive
cases. The saved weighted outcome proportion is approximately 2.20%.
This is a restricted eligible cohort, not an estimate for all US adults.

## Repository contents

| Location | Contents |
| --- | --- |
| `data/raw/` | The original 24 NHANES public-use SAS transport files; unchanged. |
| `data/nhanes_diabetes_2021_2023.csv` | Original merged table, 6,001 adults and 88 columns; retained unchanged. |
| `data/nhanes_diabetes_2021_2023_verified.csv` | Verified merged input used by both analyses; retained unchanged. |
| `scripts/nhanes_hba1c_pycharm.py` | Current Aim 1 analysis, adapted from the teammate's standalone `nhanes.py`. |
| `scripts/aim2_survey_analysis.R` | Current Aim 2 prevalence, Rao–Scott tests and adjusted logistic regression. |
| `scripts/make_figure_aim2.R` | Generates the Aim 2 prevalence/odds-ratio figure from saved CSV results. |
| `results/aim1/` | Complete saved Aim 1 run `20261102_231855_989045`, including the shared cohort manifest. |
| `results/aim2/` | Four saved Aim 2 CSV tables and `Figure_Aim2.png`. |
| `requirements.txt` | Package versions recorded with the saved Aim 1 run. |
| `docs/report_alignment.md` | Exact report replacements for the selected latest Aim 1 run. |

## Data and cohort

Source: the CDC/NCHS [NHANES August 2021–August 2023 data and documentation](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Cycle=2021-2023).
`SEQN` in the source modules corresponds to `participant_id` in the merged CSV.
Relevant codebooks are [DIQ_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DIQ_L.htm),
[GHB_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.htm) and
[DEMO_L](https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm).

`build_cohort()` in the Python script checks the supplied CSV's key eligibility,
outcome, weight and survey-design fields against those three raw XPT files.
It excludes reported diabetes/treatment, unresolved medication routing, known
pregnancy and unknown pregnancy within the public-release group. It checks
retained labels against HbA1c >=6.5%, without changing input rows or labels.
This does not independently audit every predictor derivation or establish
absence of every glucose-lowering medication or pregnancy outside the release
group. Detailed exclusions and limitations are recorded in `data_audit.json`.

The verified input SHA256 is
`e867816b31c52b97a461f035e29aefeecb17c0944589eb5477df211041a56616`.
`results/aim1/cohort_membership.csv` records inclusion/exclusion for all 6,001
input participants; Aim 2 reads this exact shared manifest.

## Run Aim 1

Python 3.10+ is required; the supplied saved run records Python **3.14.0**,
NumPy 2.3.4, pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.7 and XGBoost
3.4.1. Version differences can change numerical results despite fixed seeds.

```sh
python -m pip install -r requirements.txt
python scripts/nhanes_hba1c_pycharm.py
```

Paths are resolved from the repository root, independently of the working
directory. Optional `--csv`, `--raw-dir` and `--output` accept repository-relative
or absolute paths. The default inputs are the verified CSV and `data/raw/`.
The teammate's embedded copies of these same inputs were removed from the
repository script; the statistical analysis functions are retained.

Each run creates `results/aim1/<timestamp>/`, preserving the supplied saved
results directly under `results/aim1/`. `--quick` performs a reduced functional
check and must not supply report results. Dependencies are only installed by
the script when `--install-deps` is explicitly requested.

The analysis uses identical five outer / three inner PSU-grouped folds across
models, training-fold median/mode imputation and encoding, and WTPH2YR weights.
There is no ROS, RUS, SMOTE or additional class weighting. Hyperparameters and
the threshold targeting 80% sensitivity are selected using inner validation
only. The target does not guarantee 80% sensitivity on the outer test folds.

### Saved Aim 1 files

| File | Purpose |
| --- | --- |
| `cohort_flow.csv`, `cohort_membership.csv`, `report_numbers.csv` | Cohort flow, participant-level eligibility and headline counts. |
| `participant_characteristics.csv`, `predictor_columns.csv` | Weighted descriptive summaries and ten encoded predictor names. |
| `fold_metrics.csv`, `comparison.csv` | Per-fold results and outer-fold means/sample SDs for three models. |
| `oof_predictions.csv`, `pooled_metrics.csv` | Outer out-of-fold predictions and separately labelled pooled summaries/confusion counts. |
| `pr_curves.png` | Pooled survey-weighted precision–recall curves (report Figure 2). |
| `data_audit.json`, `requirements-analysis.txt`, `interpretation.txt` | Original run metadata, environment and interpretation limits. |

Saved result bytes are preserved from the supplied ZIP, including its original
machine paths and legacy wording in audit/log files. Those paths document the
original run and are not runtime configuration. There is no current Aim 3.

## Run Aim 2

Install R and its `survey` package. The analysis installs `survey` from CRAN if
it is absent. No R package version was recorded in the supplied package, so
an exact R-environment reproduction is not claimed.

```sh
Rscript scripts/aim2_survey_analysis.R
Rscript scripts/make_figure_aim2.R
```

The first command reads the verified CSV and saved shared Aim 1 manifest. It
builds the survey design on the full input before subsetting and writes a new
directory under `results/aim2/runs/`, preserving supplied results. The second
command, without arguments, regenerates the figure from the supplied saved
results directly under `results/aim2/`.

To analyse the membership produced by a new Aim 1 run, pass its manifest as
the first R argument. An optional second argument selects the CSV and a third
selects the output parent. For a new Aim 2 result directory, pass that directory
to the figure script:

```sh
Rscript scripts/aim2_survey_analysis.R results/aim1/<timestamp>/cohort_membership.csv
Rscript scripts/make_figure_aim2.R results/aim2/runs/<new-run-directory>
```

Replace angle-bracket placeholders with actual generated directory names.
All R paths are also resolved relative to the repository root.

Aim 2 uses age groups 18–39 / 40–59 / >=60, sex, insurance and income-to-poverty
ratio >=2 / <2 / not reported. Overall prevalence uses 5,026 participants and
126 cases. Insurance comparisons and the adjusted model exclude 12 missing
insurance responses, leaving **5,014 participants / 125 cases** for regression.
The 639 missing-income records remain a separate category. Income Rao–Scott
testing excludes that category; the regression retains it.

`Table_Aim2_final.csv` is the report-ready summary; its `p` column contains
Rao–Scott group tests. `regression_model.csv` contains adjusted odds ratios and
coefficient-specific p values. The other tables are `prevalence_by_group.csv`
and `rao_scott_tests.csv`. `Figure_Aim2.png` corresponds to report Figure 3.
The superseded trend, sensitivity and diabetes-only supplementary analyses are
not part of this simplified version.

## Results and report alignment

The latest supplied ZIP is the chosen Aim 1 source of truth. Its mean AP is
0.141 for XGB, 0.136 for RF and 0.123 for LR; LR has the highest mean ROC-AUC,
0.832. The supplied merged Word report contains results from another run.
Use [the report alignment note](docs/report_alignment.md) to update its Tables
3–5, Figure 2, numerical prose and software environment. Aim 2 values already
agree with its Table 6 and Figure 3. The Word report itself is not changed here.

Fold SD is not a confidence interval. Average precision is not accuracy or
trapezoidal PR-AUC. Pooled metrics differ from fold means. Brier skill compares
each model with an outer-training-prevalence reference, not a test-fitted risk.
These are exploratory internal-validation results without external validation
or design-based performance CIs; they do not establish overall algorithm
superiority or support clinical deployment.

Group 1 — BMI5109 Advanced Scientific Research Methods, MSc in Biomedical
Informatics, National University of Singapore.
