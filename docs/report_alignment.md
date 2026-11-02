# Alignment with the merged report

The user selected the latest teammate ZIP as the source of truth for Aim 1.
The saved run is `20261102_231855_989045`. Its original result bytes are retained.
The reviewed report is `BMI5111 Report English (merged).docx` supplied November 2.
This note lists the report changes needed; the Word document has not been edited.

The cohort (5,026 participants, 126 positive cases), nine predictors / ten encoded
columns, cohort flow and participant-characteristics values agree. Aim 2 Table 6
and Figure 3 values agree with the supplied Aim 2 results. The report's Aim 1
performance values and software environment describe a different run.

## Replace report Table 3

Outer-fold mean (sample SD), five folds; these are not pooled metrics or CIs.

| Model | AP | ROC-AUC | Brier | Brier skill |
| --- | --- | --- | --- | --- |
| LR | 0.123 (0.029) | 0.832 (0.034) | 0.0208 (0.0067) | 0.039 (0.012) |
| RF | 0.136 (0.032) | 0.819 (0.039) | 0.0208 (0.0068) | 0.041 (0.008) |
| XGB | 0.141 (0.060) | 0.828 (0.036) | 0.0208 (0.0063) | 0.034 (0.030) |

## Replace report Table 4

| Model | Sensitivity | Specificity | Precision | F1 |
| --- | --- | --- | --- | --- |
| LR | 0.787 (0.127) | 0.725 (0.057) | 0.061 (0.020) | 0.112 (0.034) |
| RF | 0.823 (0.121) | 0.649 (0.055) | 0.050 (0.016) | 0.094 (0.029) |
| XGB | 0.830 (0.121) | 0.668 (0.063) | 0.054 (0.019) | 0.101 (0.033) |

## Replace report Table 5 and its accompanying text

Unweighted confusion counts from pooled outer out-of-fold predictions.

| Model | True negative | False positive | False negative | True positive | Weighted mean predicted probability |
| --- | --- | --- | --- | --- | --- |
| LR | 3354 | 1546 | 30 | 96 | 2.16% |
| RF | 2966 | 1934 | 24 | 102 | 2.17% |
| XGB | 3065 | 1835 | 24 | 102 | 2.14% |

## Text, figure and software updates

- Abstract, Section 2.2, Discussion and Conclusion: XGB has the highest outer-fold
  mean **average precision (AP)**, 0.141; LR has the highest mean ROC-AUC, 0.832.
  AP is not accuracy. RF AP is 0.136; LR AP is 0.123.
- Mean sensitivities are 78.7%, 82.3% and 83.0% for LR, RF and XGB;
  mean precisions are 6.1%, 5.0% and 5.4%, respectively. Update related prose.
- Mean reference Brier is 0.02170; mean Brier skill is 0.039, 0.041 and 0.034
  for LR, RF and XGB. Use fold means here, not the pooled values.
- Replace Figure 2 with `results/aim1/pr_curves.png` from this same saved run.
  Figure 3 is `results/aim2/Figure_Aim2.png`. Figure 1 can use the unchanged
  cohort flow; the supplied result packages do not include its rendered diagram.
- Section 3.4: the saved Aim 1 environment is Python 3.14.0, NumPy 2.3.4,
  pandas 2.3.3, scikit-learn 1.7.2, matplotlib 3.10.7 and XGBoost 3.4.1.
  These are recorded versions, not a claim that this environment was rerun here.
- Section 3.4: cohort eligibility is now `build_cohort()` in
  `scripts/nhanes_hba1c_pycharm.py`; there is no separate `cohort.py` in this repo.
- Section 3.3.3: replace the remaining reference to "Aim 3" with "Aim 2".
- Aim 2 table/figure notes: prevalence cohort n=5,026 / 126 cases; the adjusted
  model uses n=5,014 / 125 cases after excluding 12 missing insurance responses.
  Income missingness remains a separate group (639 participants).
- The summary table's `p` column is the Rao-Scott group test. Coefficient-level
  regression p values are separately recorded in `regression_model.csv`.
  Describe aORs as adjusted odds ratios, not incidence or risk ratios.

Small performance differences do not establish algorithm superiority. No external
validation, design-based model-performance CIs or deployment model is supplied.
