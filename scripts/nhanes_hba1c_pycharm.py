#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NHANES Aim 1: three-model comparison (Python 3.10+).

Run: python scripts/nhanes_hba1c_pycharm.py
Install missing dependencies explicitly: add --install-deps.
Use --quick only for a functional check, never for report results.
Optional --csv, --raw-dir and --output paths are relative to the repository root.
Defaults: data/nhanes_diabetes_2021_2023_verified.csv, data/raw/,
and a new timestamped directory under results/aim1/.

Adapted from the teammate's November 2 standalone nhanes.py. Model fitting,
cohort rules, preprocessing, seeds and metrics are unchanged. Existing tracked
data replace the standalone file's embedded copies; source inputs are read-only.
Saved results retain their original audit metadata and recorded environment.
Aim 2 is implemented separately in aim2_survey_analysis.R.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys


def parse_cli(argv=None):
    parser = argparse.ArgumentParser(description='NHANES Aim 1 three-model analysis')
    parser.add_argument('--csv', help='Verified CSV; default data/nhanes_diabetes_2021_2023_verified.csv')
    parser.add_argument('--raw-dir', help='DIQ_L.XPT/GHB_L.XPT/DEMO_L.XPT folder for external CSV')
    parser.add_argument('--output', help='Output parent directory; default results/aim1')
    parser.add_argument('--quick', action='store_true', help='2x2 folds for debugging, NOT report results')
    parser.add_argument('--install-deps', action='store_true', help='Install missing packages with pip')
    return parser.parse_args(argv)


def ensure_dependencies(install=False):
    if sys.version_info < (3, 10):
        raise SystemExit('Please use Python 3.10 or newer (3.12 recommended).')
    packages = {'numpy':'numpy>=1.24,<3', 'pandas':'pandas>=2,<4',
                'sklearn':'scikit-learn>=1.3,<2', 'matplotlib':'matplotlib>=3.7,<4',
                'xgboost':'xgboost>=2,<4'}
    missing = [req for module,req in packages.items() if importlib.util.find_spec(module) is None]
    if missing and install:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
    elif missing:
        print('Python interpreter:', sys.executable)
        print('Missing packages:', ', '.join(missing))
        print('Run this file with --install-deps, or install using this interpreter:')
        print(subprocess.list2cmdline([sys.executable, '-m', 'pip', 'install', *missing]))
        raise SystemExit(1)
    # Version gate for preprocessing APIs; never silently change installed packages.
    from importlib.metadata import version
    pieces = version('scikit-learn').split('.')
    if tuple(int(x) for x in pieces[:2]) < (1, 3):
        raise SystemExit('scikit-learn>=1.3 required. Upgrade it in this interpreter.')


CLI = parse_cli() if __name__ == '__main__' else parse_cli([])
if __name__ == '__main__':
    ensure_dependencies(CLI.install_deps)

SCRIPT_DIRECTORY = Path(__file__).resolve().parent




"""Single eligibility definition shared by all analyses; input files are read-only.
Official rules: DIQ_L DIQ159/DIQ065 and DEMO_L RIDEXPRG.
Only key outcome/eligibility/weight/design fields are reconciled to supplied raw XPTs.
This is not an independent audit of all predictor derivations or clinical treatment.
"""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

RULE_VERSION = 'unified_hba1c_v2'
SOURCES = {
    'diabetes_questionnaire': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DIQ_L.htm',
    'demographics': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm',
    'hba1c_and_weight': 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GHB_L.htm',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_cohort(csv_path, target='diabetes_undiagnosed', weight='WTPH2YR', raw_dir=None):
    """Return cohort, ordered flow, full input ID manifest, and audit.

    DIQ010 2/3 retained, 1 excluded; 7/9/missing excluded.
    Exclude reported insulin or pills, explicit unknown medication answers.
    DIQ050 NA in non-diagnosed respondents is an official structural skip, not 'no'.
    DIQ070 expected for DIQ010=3 or DIQ160=1; require response 2 to retain.
    DIQ070 NA retained only if DIQ010=2 and DIQ160=2.
    Remaining DIQ160-unknown/unexplained pill NA excluded conservatively.
    Exclude RIDEXPRG=1 and unknown pregnancy within public-release females age20-44.
    Outside that public-release range retain NA, without certifying nonpregnancy.
    No predictor-based complete-case exclusions; training folds handle predictor NA.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    required = ['participant_id','age_years','sex','hba1c_percent',target,weight,'survey_strata','survey_psu',
                'diq010_doctor_told_diabetes','diq050_taking_insulin','diq070_taking_diabetic_pills']
    if not set(required).issubset(df):
        raise ValueError(f'Missing columns: {sorted(set(required)-set(df))}')
    if weight != 'WTPH2YR':
        raise ValueError('HbA1c analysis requires WTPH2YR.')
    if df.participant_id.isna().any() or df.participant_id.duplicated().any():
        raise ValueError('Input participant IDs must be complete and unique.')
    raw_paths = {n: (Path(raw_dir) if raw_dir else csv_path.parent / 'raw') / f'{n}.XPT' for n in ['DIQ_L','GHB_L','DEMO_L']}
    hashes = {str(csv_path.resolve()): sha256(csv_path)}
    raw = {}
    for name, path in raw_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f'Required raw file: {path}')
        hashes[str(path.resolve())] = sha256(path)
        source = pd.read_sas(path, format='xport')
        if source.SEQN.duplicated().any():
            raise ValueError(f'Duplicate SEQN in {name}')
        if not df.participant_id.isin(source.SEQN).all():
            raise ValueError(f'Unmatched input IDs in {name}')
        raw[name] = source.set_index('SEQN').reindex(df.participant_id).reset_index(drop=True)
    checks = {}
    mapping = {
        'diq010_doctor_told_diabetes': ('DIQ_L','DIQ010'),
        'diq050_taking_insulin': ('DIQ_L','DIQ050'),
        'diq070_taking_diabetic_pills': ('DIQ_L','DIQ070'),
        'hba1c_percent': ('GHB_L','LBXGH'), 'WTPH2YR': ('GHB_L','WTPH2YR'),
        'age_years': ('DEMO_L','RIDAGEYR'), 'survey_strata': ('DEMO_L','SDMVSTRA'),
        'survey_psu': ('DEMO_L','SDMVPSU')}
    for col, (file, field) in mapping.items():
        ok = np.isclose(pd.to_numeric(df[col], errors='raise'), raw[file][field], equal_nan=True)
        checks[col] = int((~ok).sum())
        if not ok.all():
            raise ValueError(f'{col}: {checks[col]} records differ from raw {field}; stop, no repair.')
    raw_sex = raw['DEMO_L'].RIAGENDR.map({1:'male', 2:'female'})
    checks['sex'] = int((~df.sex.fillna('NA').eq(raw_sex.fillna('NA'))).sum())
    if checks['sex']:
        raise ValueError('Sex differs from raw RIAGENDR.')
    q = raw['DIQ_L'].DIQ010
    insulin = raw['DIQ_L'].DIQ050
    pills = raw['DIQ_L'].DIQ070
    pre = raw['DIQ_L'].DIQ160
    age = raw['DEMO_L'].RIDAGEYR
    preg = raw['DEMO_L'].RIDEXPRG
    release_eligible = raw['DEMO_L'].RIAGENDR.eq(2) & age.between(20,44)
    for name, s, allowed in [('DIQ010',q,[1,2,3,7,9]),('DIQ050',insulin,[1,2,7,9]),
                             ('DIQ070',pills,[1,2,7,9]),('DIQ160',pre,[1,2,7,9]),('RIDEXPRG',preg,[1,2,3])]:
        if not s.dropna().isin(allowed).all():
            raise ValueError(f'Unexpected code in {name}')
    alive = pd.Series(True, index=df.index)
    reason = pd.Series('included', index=df.index)
    flow = [{'stage':'input_merged_adults','excluded':0,'remaining':len(df)}]
    def exclude(name, mask):
        nonlocal alive
        hit = alive & mask
        reason.loc[hit] = name
        alive &= ~mask
        flow.append({'stage':name, 'excluded':int(hit.sum()), 'remaining':int(alive.sum())})
    exclude('under18_or_missing_age', ~age.ge(18))
    exclude('self_reported_diabetes', q.eq(1))
    exclude('diagnosis_unknown_or_missing', ~q.isin([2,3]))
    exclude('reported_diabetes_treatment', insulin.eq(1) | pills.eq(1))
    exclude('medication_refused_or_unknown', insulin.isin([7,9]) | pills.isin([7,9]))
    expected_pills = q.eq(3) | pre.eq(1)
    exclude('missing_pills_when_expected', expected_pills & pills.isna())
    structural_pills_skip = q.eq(2) & pre.eq(2) & pills.isna()
    exclude('unresolved_pills_skip', pills.isna() & ~expected_pills & ~structural_pills_skip)
    exclude('known_pregnancy', preg.eq(1))
    exclude('pregnancy_unknown_in_release_group', release_eligible & ~preg.isin([1,2]))
    hba = pd.to_numeric(df.hba1c_percent, errors='raise')
    exclude('hba1c_missing_or_invalid', ~np.isfinite(hba) | hba.le(0))
    y = pd.to_numeric(df[target], errors='raise')
    exclude('outcome_missing', y.isna())
    w = pd.to_numeric(df[weight], errors='raise')
    exclude('weight_missing_nonpositive_or_invalid', ~np.isfinite(w) | w.le(0))
    exclude('survey_design_missing', df[['survey_strata','survey_psu']].isna().any(axis=1))
    if not y[alive].isin([0,1]).all() or not y[alive].eq(hba[alive].ge(6.5).astype(int)).all():
        raise ValueError('Retained labels disagree with HbA1c definition; stop, do not overwrite labels.')
    membership = pd.DataFrame({
        'participant_id':df.participant_id, 'included':alive.astype(int), 'first_exclusion_reason':reason,
        'diq010_raw':q, 'diq160_raw':pre, 'diq050_raw':insulin, 'diq070_raw':pills,
        'pregnancy_raw':preg,
        'pregnancy_release_eligible':release_eligible.astype(int),
        'oral_medication_status':np.select([pills.eq(1), pills.eq(2), structural_pills_skip],
                                         ['reported_yes','reported_no','documented_structural_skip'],default='unknown'),
    })
    cohort = df.loc[alive].copy().reset_index(drop=True)
    cohort['eligibility_diq160'] = pre[alive].to_numpy()
    cohort['eligibility_pregnancy_raw'] = preg[alive].to_numpy()
    audit = {
        'rule_version':RULE_VERSION, 'input_rows':len(df), 'analysis_rows':len(cohort),
        'positive':int(y[alive].sum()), 'negative':int((y[alive]==0).sum()),
        'unweighted_prevalence':float(y[alive].mean()),
        'weighted_prevalence':float(np.average(y[alive],weights=w[alive])),
        'exclusions_in_order':{r['stage']:r['excluded'] for r in flow[1:]},
        'raw_key_field_mismatch_counts':checks,
        'oral_medication_structural_skip_retained_n':int((alive & structural_pills_skip).sum()),
        'insulin_structural_skip_retained_n':int((alive & insulin.isna()).sum()),
        'outside_pregnancy_public_release_retained_n':int((alive & ~release_eligible & preg.isna()).sum()),
        'borderline_diagnosis_retained_n':int((alive & q.eq(3)).sum()),
        'membership_sha256':hashlib.sha256(','.join(map(str,sorted(cohort.participant_id.astype(int)))).encode()).hexdigest(),
        'source_file_sha256':hashes, 'official_codebooks':SOURCES,
        'source_scope':'Key eligibility/outcome/weight/design fields reconciled to supplied raw XPT files. Predictor derivations and file authenticity not independently audited.',
        'medication_limit':'Insulin not asked without diagnosed diabetes; structural skips do not establish absence of all treatment. Oral-agent question does not cover every glucose-lowering drug.',
        'pregnancy_limit':'Excludes known pregnancy and unknown status in public-release females20-44; outside-release missing values retained, pregnancy cannot be ruled out.',
        'no_source_rows_or_labels_overwritten':True,
    }
    return cohort, pd.DataFrame(flow), membership, audit

# ==================== 用户设置：通常只需要修改这里 ====================
MODELS = ["lr", "rf", "xgb"]                  # lr逻辑回归，rf随机森林，xgb XGBoost
QUICK_RUN = False                            # True=快速检查；False=完整比较
OUTER_FOLDS = 5
INNER_FOLDS = 3
RECALL_TARGET = 0.80                          # 内部验证上的目标敏感度
RANDOM_SEED = 2026
N_JOBS = 2                                   # 并行线程数
TARGET_COLUMN = "diabetes_undiagnosed"        # 使用新CSV中已重算的标签；代码只核对，不改写
WEIGHT_COLUMN = "WTPH2YR"                    # 必须存在且为有效正数
EXTRA_DROP_COLUMNS = []                      # 如有其他泄露变量，在这里添加列名
# =====================================================================

from types import SimpleNamespace
import importlib.metadata
import json
import hashlib
from datetime import datetime
from pathlib import Path
import warnings
import platform

REPO_ROOT = SCRIPT_DIRECTORY.parent

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 连续变量保留原量纲；性别二元编码，吸烟设never参考组。
NUMERIC_PREDICTORS = ['age_years', 'waist_to_height_ratio',
                      'avg_systolic_bp_mmhg', 'hdl_cholesterol_mg_dl',
                      'total_cholesterol_mg_dl', 'family_income_poverty_ratio',
                      'has_health_insurance']
CATEGORICAL_LEVELS = {'sex': ['female', 'male'],
                      'smoking_status': ['never', 'former', 'current']}
PREDICTORS = NUMERIC_PREDICTORS + list(CATEGORICAL_LEVELS)
ENCODED_COLUMNS = NUMERIC_PREDICTORS + ['sex_male', 'smoking_status_former', 'smoking_status_current']

TARGETS = {'diabetes', 'diabetes_diagnosed', 'diabetes_undiagnosed', 'glycemic_status'}
DESIGN = {'participant_id', 'survey_cycle', 'survey_strata', 'survey_psu',
          'SDMVSTRA', 'SDMVPSU', 'SEQN', 'WTPH2YR', 'WTMEC2YR', 'WTINT2YR', 'WTSAF2YR'}
FASTING = {'ldl_cholesterol_mg_dl', 'triglycerides_mg_dl', 'triglyceride_hdl_ratio'}
# 明确按类别处理的数字编码；其余字符串列也自动作为类别。
CATEGORIES = {'age_group_brfss', 'education_code', 'income_poverty_category',
              'general_health', 'routine_healthcare_place'}
# 避免同一信息的数值编码/文字标签重复进入模型。
REDUNDANT = {'age_group_brfss', 'education_code', 'general_health_label',
             'bmi_category', 'income_poverty_category',
             'pregnant'}   # 孕妇已整体排除，该列不再作为预测变量


def recorded_path(path):
    """仓库内路径在审计中保存为相对路径；外部绝对路径保持其实际含义。"""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def settings_to_args():
    data_path = resolve_path(CLI.csv or 'data/nhanes_diabetes_2021_2023_verified.csv')
    if not data_path.is_file():
        raise FileNotFoundError(f'CSV not found: {data_path}')
    raw_dir = resolve_path(CLI.raw_dir) if CLI.raw_dir else data_path.parent / 'raw'
    output = resolve_path(CLI.output or 'results/aim1')
    output.mkdir(parents=True, exist_ok=True)
    print('Interpreter:', sys.executable, flush=True)
    print('Input CSV:', data_path, flush=True)
    return SimpleNamespace(csv=str(data_path), raw_dir=str(raw_dir), out=str(output),
                           target=TARGET_COLUMN, weight_col=WEIGHT_COLUMN,
                           models=list(MODELS), outer_folds=OUTER_FOLDS,
                           inner_folds=INNER_FOLDS, recall_target=RECALL_TARGET,
                           seed=RANDOM_SEED, jobs=N_JOBS, quick=CLI.quick or QUICK_RUN,
                           extra_drop=list(EXTRA_DROP_COLUMNS))


def load_data(args):
    # 输入文件只读；保留哈希值，以便运行后核实没有被修改。
    source_hash = hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()
    df, flow, membership, cohort_audit = build_cohort(args.csv, args.target, args.weight_col, args.raw_dir)
    y = df[args.target].to_numpy(dtype=int)
    w = df[args.weight_col].to_numpy(dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError('Retained cohort must contain both outcomes.')
    wc = args.weight_col
    input_n = cohort_audit['input_rows']
    exclusions = cohort_audit['exclusions_in_order']
    groups = (df.survey_strata.astype(str) + '_' + df.survey_psu.astype(str)).to_numpy()
    excluded_cols = TARGETS | DESIGN | FASTING | REDUNDANT | {args.target, wc} | set(args.extra_drop)
    excluded_cols |= {c for c in df if c.startswith('survey_weight') or c.upper().startswith('WT')}
    # 诊断变量通常在另一个 audit 文件；即便合并回来也不得进入特征。
    excluded_cols |= {c for c in df if any(t in c.lower() for t in
                        ['hba1c', 'glucose', 'glycemic', 'insulin', 'homa', 'diq', 'lbxgh', 'lbxglu', 'diabetes'])}
    missing_predictors = set(PREDICTORS) - set(df.columns)
    if missing_predictors:
        raise ValueError(f'缺少预先指定变量：{sorted(missing_predictors)}')
    if set(PREDICTORS) & excluded_cols:
        raise ValueError('预先指定预测变量与禁止使用字段冲突。')
    X = df[PREDICTORS].copy()
    cats = list(CATEGORICAL_LEVELS)
    for c, levels in CATEGORICAL_LEVELS.items():
        if not X[c].dropna().isin(levels).all():
            raise ValueError(f'{c}出现未知类别；请检查数据字典。')
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    for c in NUMERIC_PREDICTORS:
        X[c] = pd.to_numeric(X[c], errors='raise').replace([np.inf, -np.inf], np.nan)
    if not X['has_health_insurance'].dropna().isin([0, 1]).all():
        raise ValueError('保险变量必须为0/1或缺失。')
    if not np.isfinite(X['age_years']).all():
        raise ValueError('年龄必须完整。')
    audit = dict(status='HBA1C_COHORT_INTERNAL_CHECKS_PASSED', quick_check_only=args.quick,
                 input_rows=input_n, exclusions_in_order=exclusions,
                 analysis_rows=len(y), positive=int(y.sum()), negative=int((y==0).sum()),
                 unweighted_prevalence=float(y.mean()), weighted_prevalence=float(np.average(y,weights=w)),
                 target=args.target, weight_column=wc, n_psu_groups=len(np.unique(groups)),
                 predictors=list(X.columns), categorical=cats,
                 raw_predictor_count=len(PREDICTORS),
                 encoded_predictor_count=len(ENCODED_COLUMNS),
                 encoded_columns=ENCODED_COLUMNS,
                 predictor_missing_counts=X.isna().sum().to_dict(),
                 imbalance_strategy='survey weights only; no resampling or extra class weights',
                 predictor_selection='Fixed reduced set before this rerun; not selected using outcome associations. Revision follows review of earlier analyses, not an independent prospective validation.',

                 excluded_columns=sorted(set(df.columns)-set(X.columns)),
                 input_sha256=source_hash,
                 cohort_rules=cohort_audit,
                 no_labels_or_source_rows_written=True,
                 path_base='repository_root',
                 options={**vars(args), 'csv': recorded_path(args.csv),
                          'out': recorded_path(args.out)},
                 versions={m:importlib.metadata.version(m) for m in
                    ['numpy','pandas','scikit-learn','matplotlib']})
    return df, X, y, w, groups, cats, audit, flow, membership


def make_splits(y, groups, n, seed):
    if len(np.unique(groups)) < n:
        raise ValueError('独立 PSU 数不足，请减少折数。')
    cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed)
    splits = list(cv.split(np.zeros(len(y)), y, groups))
    for tr, te in splits:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            raise ValueError('某折只有一个类别，请减少折数并检查 PSU 的阳性分布。')
        assert not set(groups[tr]) & set(groups[te])
    return splits


def preprocessor(X, cats):
    nums = [c for c in X if c not in cats]
    return ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=False,
                                                  keep_empty_features=True)),
                          ('scale', StandardScaler())]), nums),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent',
                                                  add_indicator=False, keep_empty_features=True)),
                          ('encode', OneHotEncoder(categories=[CATEGORICAL_LEVELS[c] for c in cats], drop='first', handle_unknown='error', sparse_output=False))]), cats)
    ], sparse_threshold=0)


def configs(model, quick=False):
    # 每种算法只调少量参数；不再比较平衡方法。
    hyper = {'lr': [{'C': 0.1}, {'C': 1.0}],
             'rf': [{'max_depth': 3, 'min_samples_leaf': 20},
                    {'max_depth': 5, 'min_samples_leaf': 30}],
             'xgb': [{'max_depth': 2, 'min_child_weight': 5},
                     {'max_depth': 3, 'min_child_weight': 10}]}[model]
    return [dict(model=model, hyper=h) for h in (hyper[:1] if quick else hyper)]


def estimator(config, args, seed):
    h = config['hyper']
    trees = 25 if args.quick else 150
    if config['model'] == 'lr':
        return LogisticRegression(C=h['C'], solver='lbfgs', max_iter=2500)
    if config['model'] == 'rf':
        return RandomForestClassifier(**h, n_estimators=trees, max_features='sqrt',
                                      random_state=seed, n_jobs=args.jobs)
    from xgboost import XGBClassifier
    return XGBClassifier(**h, n_estimators=trees, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, reg_lambda=2.0, tree_method='hist',
                         objective='binary:logistic', eval_metric='logloss',
                         scale_pos_weight=1, random_state=seed, n_jobs=args.jobs)


def predict_fit(config, A, y, w, B, args, seed):
    # 仅按常数将训练权重均值归一为1；不改变相对调查权重。
    fit_w = np.asarray(w, dtype=float) * len(y) / np.sum(w)
    model = estimator(config, args, seed)
    model.fit(A, y, sample_weight=fit_w)
    return model.predict_proba(B)[:, 1]


def tune(model, prepared, y, w, args, seed):
    best = None
    for config in configs(model, args.quick):
        oof = np.full(len(y), np.nan)
        scores = []
        for k, (tr, va, A, B) in enumerate(prepared):
            p = predict_fit(config, A, y[tr], w[tr], B, args, seed + k)
            oof[va] = p
            scores.append(average_precision_score(y[va], p, sample_weight=w[va]))
        score = float(np.mean(scores))
        assert np.isfinite(oof).all()
        if best is None or score > best['inner_ap']:
            best = dict(config=config, inner_ap=score, oof=oof)
    return best


def threshold_at_recall(y, p, w, target):
    _, recall, thresholds = precision_recall_curve(y, p, sample_weight=w)
    valid = np.flatnonzero(recall[:-1] >= target)
    # 最大阈值：在内部OOF达到召回目标的阈值中，尽量减少预测阳性人数。
    return float(thresholds[valid[-1]]) if len(valid) else 0.0


def metrics(y, p, w, predicted, reference_probability):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1], sample_weight=w).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    brier = float(np.average((y - p) ** 2, weights=w))
    reference_brier = float(np.average((y - reference_probability) ** 2, weights=w))
    skill = 1 - brier / reference_brier if reference_brier > 0 else float('nan')
    return dict(AP=float(average_precision_score(y, p, sample_weight=w)),
                ROC_AUC=float(roc_auc_score(y, p, sample_weight=w)),
                sensitivity=float(recall), specificity=float(tn / (tn + fp)),
                precision=float(precision), F1=float(2 * precision * recall / (precision + recall))
                if precision + recall else 0.0,
                Brier=brier, Brier_reference=reference_brier, Brier_skill=skill,
                weighted_prevalence=float(np.average(y, weights=w)))


def plots(oof, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, d in oof.groupby('combination'):
        precision, recall, _ = precision_recall_curve(d.y, d.probability, sample_weight=d.survey_weight)
        ax.plot(recall, precision, label=name.upper())
    base = oof[oof.combination == oof.combination.iloc[0]]
    ax.axhline(np.average(base.y, weights=base.survey_weight), color='gray', linestyle='--', label='Weighted prevalence')
    ax.set(xlabel='Recall', ylabel='Precision', title='Survey-weighted PR curves (out-of-fold)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / 'pr_curves.png', dpi=180)
    plt.close(fig)


def export_cohort_tables(df, X, y, w, audit, out, flow, membership):
    flow.to_csv(out / 'cohort_flow.csv', index=False)
    membership.to_csv(out / 'cohort_membership.csv', index=False)
    pd.DataFrame([{
        'analysis_n': len(y), 'positive_n': int(y.sum()), 'negative_n': int((y == 0).sum()),
        'unweighted_prevalence': y.mean(), 'weighted_prevalence': np.average(y, weights=w),
        'raw_predictors': len(PREDICTORS), 'encoded_predictors': len(ENCODED_COLUMNS),
        'weight': audit['weight_column']
    }]).to_csv(out / 'report_numbers.csv', index=False)
    pd.DataFrame({'encoded_column': ENCODED_COLUMNS}).to_csv(out / 'predictor_columns.csv', index=False)
    rows = []
    for c in PREDICTORS:
        for label, mask in [('All', np.ones(len(y), dtype=bool)), ('Outcome 0', y == 0), ('Outcome 1', y == 1)]:
            valid = mask & X[c].notna().to_numpy()
            if c in CATEGORICAL_LEVELS or c == 'has_health_insurance':
                levels = CATEGORICAL_LEVELS.get(c, [0, 1])
                for level in levels:
                    hits = valid & X[c].eq(level).to_numpy()
                    rows.append(dict(variable=c, group=label, statistic=f'proportion {level}',
                                     n=int(hits.sum()), missing_n=int((mask & ~valid).sum()),
                                     weighted_value=float(w[hits].sum() / w[valid].sum())))
            else:
                v = X.loc[valid, c].to_numpy(dtype=float)
                mean = np.average(v, weights=w[valid])
                sd = np.sqrt(np.average((v - mean) ** 2, weights=w[valid]))
                rows.append(dict(variable=c, group=label, statistic='mean', n=int(valid.sum()),
                                 missing_n=int((mask & ~valid).sum()), weighted_value=mean,
                                 weighted_descriptive_sd=sd))
    pd.DataFrame(rows).to_csv(out / 'participant_characteristics.csv', index=False)


def main():
    args = settings_to_args()
    if not 0 < args.recall_target < 1:
        raise ValueError('RECALL_TARGET 必须在0与1之间。')
    args.models = list(dict.fromkeys(args.models))
    if 'xgb' in args.models:
        try:
            import xgboost  # noqa: F401
        except ImportError as exc:
            raise SystemExit('缺少 xgboost：先运行 python -m pip install xgboost；或将顶部 MODELS 改为 ["lr", "rf"]。') from exc
    if args.quick:
        args.outer_folds, args.inner_folds = 2, 2
    if min(args.outer_folds, args.inner_folds) < 2:
        raise ValueError('交叉验证至少2折。')
    out = Path(args.out) / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    out.mkdir(parents=True, exist_ok=False)
    df, X, y, w, groups, cats, audit, flow, membership = load_data(args)
    audit['output_directory'] = recorded_path(out)
    audit['python_version'] = platform.python_version()
    audit['evaluation_version'] = 'brier_training_prevalence_reference_v1'
    audit['brier_reference'] = 'Each outer test participant receives the weighted outcome prevalence of that outer training set. BSS=1-model_Brier/reference_Brier. No test labels used to fit reference.'
    audit['full_cohort_constant_risk_descriptive'] = float(np.average(y, weights=w))
    audit['full_cohort_constant_brier_descriptive'] = float(np.average(y, weights=w) * (1 - np.average(y, weights=w)))
    if 'xgb' in args.models:
        audit['versions']['xgboost'] = importlib.metadata.version('xgboost')
    (out / 'requirements-analysis.txt').write_text('\n'.join(f'{k}=={v}' for k, v in audit['versions'].items()) + '\n', encoding='utf-8')
    export_cohort_tables(df, X, y, w, audit, out, flow, membership)
    (out / 'data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"{audit['status']} | n={len(y)}, positive={y.sum()}, weight={audit['weight_column']}", flush=True)
    print('排除记录：' + json.dumps(audit['exclusions_in_order'], ensure_ascii=False), flush=True)
    print('输入CSV/XPT只读；关键字段已逐人核对；统一入排规则见本文件build_cohort()，详细限制见data_audit.json。', flush=True)
    outer = make_splits(y, groups, args.outer_folds, args.seed)
    rows, predictions = [], []
    for fold, (tr, te) in enumerate(outer, start=1):
        print(f'外层 {fold}/{len(outer)}；训练阳性 {y[tr].sum()}，测试阳性 {y[te].sum()}', flush=True)
        Xtr = X.iloc[tr].reset_index(drop=True)
        yt, wt = y[tr], w[tr]
        reference_risk = float(np.average(yt, weights=wt))
        inner = make_splits(yt, groups[tr], args.inner_folds, args.seed + fold)
        # 相同内部划分与预处理供所有模型共享，绝不使用外层测试信息。
        prepared = []
        for it, iv in inner:
            prep = preprocessor(Xtr, cats)
            A = prep.fit_transform(Xtr.iloc[it])
            B = prep.transform(Xtr.iloc[iv])
            prepared.append((it, iv, A, B))
        outer_prep = preprocessor(Xtr, cats)
        A = outer_prep.fit_transform(Xtr)
        B = outer_prep.transform(X.iloc[te])
        if A.shape[1] != len(ENCODED_COLUMNS):
            raise ValueError(f'编码后实际列数为{A.shape[1]}，预期{len(ENCODED_COLUMNS)}。')
        for model in args.models:
            best = tune(model, prepared, yt, wt, args, args.seed + fold * 100)
            threshold = threshold_at_recall(yt, best['oof'], wt, args.recall_target)
            p = predict_fit(best['config'], A, yt, wt, B, args, args.seed + fold * 100)
            pred = (p >= threshold).astype(int)
            row = dict(fold=fold, combination=model, model=model,
                       inner_AP=best['inner_ap'], threshold=threshold, reference_probability=reference_risk, n_train=len(tr),
                       positives_train=int(yt.sum()), n_test=len(te),
                       positives_test=int(y[te].sum()), encoded_predictors=A.shape[1],
                       config=json.dumps(best['config']), **metrics(y[te], p, w[te], pred, reference_risk))
            rows.append(row)
            predictions.append(pd.DataFrame(dict(
                participant_id=df.participant_id.iloc[te].to_numpy(), fold=fold,
                combination=model, y=y[te], probability=p, predicted=pred,
                threshold=threshold, reference_probability=reference_risk, survey_weight=w[te], psu_group=groups[te])))
            print(f"  {model:5s} inner AP={best['inner_ap']:.4f}, outer AP={row['AP']:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(out / 'fold_metrics.csv', index=False)
    result = pd.DataFrame(rows)
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_csv(out / 'oof_predictions.csv', index=False)
    keys = ['AP', 'ROC_AUC', 'sensitivity', 'specificity', 'precision', 'F1', 'Brier', 'Brier_reference', 'Brier_skill']
    summary = result.groupby('combination')[keys].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.sort_values('AP_mean', ascending=False)
    summary.insert(0, 'descriptive_rank', np.arange(1, len(summary) + 1))
    summary.to_csv(out / 'comparison.csv')
    plots(oof, out)
    pooled = []
    for name, d in oof.groupby('combination'):
        tn, fp, fn, tp = confusion_matrix(d.y, d.predicted, labels=[0, 1]).ravel()
        pooled.append(dict(model=name, TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
                           weighted_mean_predicted_probability=float(np.average(d.probability, weights=d.survey_weight)),
                           **metrics(d.y.to_numpy(), d.probability.to_numpy(), d.survey_weight.to_numpy(), d.predicted.to_numpy(), d.reference_probability.to_numpy())))
    pd.DataFrame(pooled).to_csv(out / 'pooled_metrics.csv', index=False)
    explanation = (
        f"状态：{audit['status']}；quick={args.quick}\n"
        f"分析人群：{len(y)}人，{int(y.sum())}阳性；未加权比例{y.mean():.4%}；调查加权比例{np.average(y, weights=w):.4%}。\n"
        f"原始预测变量{len(PREDICTORS)}个，编码后{len(ENCODED_COLUMNS)}列。\n"
        '只比较LR、RF、XGB；无ROS/RUS/SMOTE或额外类别权重。\n'
        'comparison.csv为外层折均值和SD；SD不是置信区间。pooled_metrics.csv为池化OOF指标，二者不必相同。\n'
        '外层排名仅作描述；最高分模型并没有获得独立的最终验证，小差异不代表显著胜出。\n'
        '阈值仅使用内层OOF确定；外层敏感度不保证达到80%。AP为average precision。\n'
        'Brier_reference为各外层训练集加权患病率常数预测的测试Brier；Brier_skill=1-Brier/Brier_reference，接近0表示改善有限。\n'
        '全队列患病率p的p(1-p)仅是描述性基线，不用于拟合测试参考预测。\n'
        'PSU分组CV不是完整设计型推断；没有设计型CI或外部验证。\n'
        '关键字段已核对上传原XPT；不代表所有用药或妊娠状态都可获知；详见审计。\n'
        'Aim 2关联分析由aim2_survey_analysis.R执行；本脚本不执行SHAP或最终部署模型训练。\n'
        'QUICK_RUN=True仅用于功能检查，不能放入最终报告。\n'
    )
    (out / 'interpretation.txt').write_text(explanation, encoding='utf-8')
    print('\n' + explanation)
    source_checks = {p: sha256(p) == h for p, h in audit['cohort_rules']['source_file_sha256'].items()}
    audit['source_files_unchanged_after_run'] = source_checks
    unchanged = all(source_checks.values())
    audit['input_unchanged_after_run'] = unchanged
    (out / 'data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'输入CSV未改变：{unchanged}')
    print(f'结果目录：{out.resolve()}')
    return out, args.csv, args.quick





if __name__ == '__main__':
    main()
