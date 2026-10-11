#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""NHANES：三个模型 × 四种不平衡方法（Python 3.10+）。

PyCharm 使用方法：
1. 将本 .py 文件与 CSV 放在同一个文件夹。
2. 在 PyCharm 当前解释器中安装 numpy、pandas、scikit-learn>=1.3、matplotlib、xgboost。
3. 右键本文件 → Run，或点击绿色运行按钮。无需设置运行参数。
4. 结果保存在本文件所在目录下的 results_balance_hba1c 文件夹。
只需修改下面“用户设置”中的变量。DATA_FILE 也支持绝对路径，例如
r"C:\Users\YourName\Desktop\nhanes_diabetes_2021_2023_verified.csv"。
本文件独立运行，不依赖之前的命令行版脚本。

本版适配用户提供的 verified CSV：仅读取该文件，不修改或重新保存CSV。
依据诊断/用药字段筛选分析对象，排除明确的拒答或不知道，不把问卷跳题
空值全部当成未服药。非确诊者胰岛素问题空值按调查跳题保留；DIQ010=2
且口服药字段空值、又缺少DIQ160时，按提供文件的跳题解释暂予保留，并在
audit中明确计数。未用原始XPT独立验证来源；不等于已确认没有任何降糖治疗。
现有阳性标签在保留人群中必须与HbA1c>=6.5一致，否则停止，不自动改标签。
仅使用WTPH2YR，不回退到MEC权重。另排除孕妇（与 Aim 3 统一）。按本次上传数据应得到5058人/128阳性。
每次输出保存在新建的时间戳子目录，避免覆盖之前结果。

主要输出（默认 results_balance_hba1c/）：
  data_audit.json：实际使用的标签、权重、特征、样本数、软件版本。
  fold_metrics.csv：每个外层折、每个组合的表现及内部选中的参数/阈值。
  comparison.csv：外层平均指标、折间标准差（不是标准误/置信区间）。
  oof_predictions.csv：原始测试对象的预测和调查权重，便于后续审计。
  nested_selection_metrics.csv：在内部选择“模型+平衡方法”的整个流程表现。
  pr_curves.png、calibration.png：池化 OOF 描述图；原始概率，未再校准。
  interpretation.txt：排名的解释及限制。

设计：外层默认5折、内层3折（INNER_FOLDS=5 可改）；按 strata+PSU 分组，
同一 PSU 不跨训练/测试，以结局尽量平衡各折。这是 PSU 分组内部验证，
不等价于完整设计型推断；不计算设计型 CI，也不是外部验证。
插补、编码、标准化仅用训练折；ROS/RUS 在预处理后、模型拟合前进行。
训练调查权重乘类别代价；评估只用调查权重。所有权重仅按常数归一化，
不截断；不把权重、设计变量、其他结局或诊断血糖指标当作预测特征。
默认删除 LDL/甘油三酯及其比值，以免把空腹子样本设计缺失直接普通插补。
自动挑选阈值仅用内部 OOF 概率（默认目标敏感度0.8），外层不调阈值。
比较表排名是探索性描述；最高外层分数有选择乐观偏差，不能直接叫最终
模型的无偏测试表现。nested_selection_metrics.csv 评估内部选优流程。
仅实现 A-D；不包含 SMOTENC、SHAP、MICE 或 Aim 3 调查加权关联分析。

文档：
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
https://xgboost.readthedocs.io/en/stable/python/python_api.html
"""

# ==================== 用户设置：通常只需要修改这里 ====================
DATA_FILE = "nhanes_diabetes_2021_2023_verified.csv"  # CSV 文件名，或 r"C:\...\data.csv"
OUTPUT_FOLDER = "results_balance_hba1c"            # 结果文件夹，可填写绝对路径
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

METHODS = ['baseline', 'class_weight', 'ros', 'rus']
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


def settings_to_args():
    """所有相对路径以脚本所在目录为准，不依赖PyCharm的Working directory。"""
    base = Path(__file__).resolve().parent
    data_path = Path(DATA_FILE).expanduser()
    if not data_path.is_absolute():
        data_path = base / data_path
    if not data_path.is_file():
        raise FileNotFoundError(
            f"找不到数据文件：{data_path}\n"
            "请把 CSV 与本脚本放在同一个文件夹，或修改顶部 DATA_FILE 为实际完整路径。"
        )
    output_path = Path(OUTPUT_FOLDER).expanduser()
    if not output_path.is_absolute():
        output_path = base / output_path
    if not MODELS or any(m not in ['lr', 'rf', 'xgb'] for m in MODELS):
        raise ValueError('MODELS 必须包含 lr、rf、xgb 中的至少一个模型。')
    return SimpleNamespace(csv=str(data_path), out=str(output_path),
                           target=TARGET_COLUMN, weight_col=WEIGHT_COLUMN,
                           models=list(MODELS),
                           outer_folds=OUTER_FOLDS, inner_folds=INNER_FOLDS,
                           recall_target=RECALL_TARGET, seed=RANDOM_SEED,
                           jobs=N_JOBS, quick=QUICK_RUN, extra_drop=list(EXTRA_DROP_COLUMNS))


def load_data(args):
    # 输入文件只读；保留哈希值，以便运行后核实没有被修改。
    source_hash = hashlib.sha256(Path(args.csv).read_bytes()).hexdigest()
    df = pd.read_csv(args.csv)
    qcol, icol, pcol = ('diq010_doctor_told_diabetes', 'diq050_taking_insulin',
                        'diq070_taking_diabetic_pills')
    needed = [args.target, 'participant_id', 'age_years', 'survey_strata', 'survey_psu',
              'hba1c_percent', qcol, icol, pcol, args.weight_col]
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f'缺少列：{sorted(missing)}；请使用新版 verified CSV。')
    if df.participant_id.isna().any() or df.participant_id.duplicated().any():
        raise ValueError('participant_id 缺失或重复。')
    input_n = len(df)
    q = pd.to_numeric(df[qcol], errors='raise')
    insulin = pd.to_numeric(df[icol], errors='raise')
    pills = pd.to_numeric(df[pcol], errors='raise')
    for name, vals, allowed in [(qcol,q,{1,2,3,7,9}), (icol,insulin,{1,2,7,9}), (pcol,pills,{1,2,7,9})]:
        if not vals.dropna().isin(allowed).all():
            raise ValueError(f'{name} 存在未识别编码。')
    alive = pd.Series(True, index=df.index)
    exclusions = {}
    def exclude(reason, mask):
        nonlocal alive
        removed = alive & mask
        exclusions[reason] = int(removed.sum())
        alive = alive & ~mask
    exclude('under18_or_missing_age', ~pd.to_numeric(df.age_years, errors='raise').ge(18))
    exclude('self_reported_diagnosis', q.eq(1))
    exclude('unknown_diagnosis', ~q.isin([2,3]))
    exclude('reported_treatment_without_diagnosis', insulin.eq(1) | pills.eq(1))
    exclude('explicit_unknown_or_refused_medication', insulin.isin([7,9]) | pills.isin([7,9]))
    # 与 Aim 3 统一：排除孕妇（妊娠影响 HbA1c，结局判定不可靠）
    if 'pregnant' not in df:
        raise ValueError('缺少 pregnant 列；请使用新版 verified CSV。')
    exclude('pregnant', pd.to_numeric(df['pregnant'], errors='raise').eq(1))
    # Borderline受试者按路由应回答口服药问题；这里的空值不当成跳题。
    exclude('missing_pills_when_question_expected', q.eq(3) & pills.isna())
    if 'DIQ160' in df or 'diq160_prediabetes' in df:
        pc = 'DIQ160' if 'DIQ160' in df else 'diq160_prediabetes'
        prediab = pd.to_numeric(df[pc], errors='raise')
        exclude('missing_pills_with_reported_or_unknown_prediabetes',
                q.eq(2) & pills.isna() & ~prediab.eq(2))
        unresolved_skip = 0
    else:
        unresolved_skip = int((alive & q.eq(2) & pills.isna()).sum())
    hba = pd.to_numeric(df.hba1c_percent, errors='raise')
    exclude('missing_hba1c', hba.isna())
    target = pd.to_numeric(df[args.target], errors='raise')
    exclude('missing_outcome', target.isna())
    if not target[alive].isin([0,1]).all():
        raise ValueError('结局需为0/1。')
    if not np.isfinite(hba[alive]).all() or (hba[alive] <= 0).any():
        raise ValueError('HbA1c存在非正数或无穷值。')
    mismatch = target[alive].astype(int).ne(hba[alive].ge(6.5).astype(int))
    if mismatch.any():
        raise ValueError(f'保留人群中有{int(mismatch.sum())}个标签与HbA1c规则不符；停止，不自动重标记。')
    df = df.loc[alive].copy().reset_index(drop=True)
    y = pd.to_numeric(df[args.target], errors='raise').to_numpy(dtype=int)
    if set(np.unique(y)) != {0,1}:
        raise ValueError('保留人群必须同时包含阳性和阴性。')
    wc = args.weight_col
    if wc != 'WTPH2YR':
        raise ValueError('本版为HbA1c主分析，请使用WTPH2YR。')
    w = pd.to_numeric(df[wc], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(w).all() or (w <= 0).any():
        raise ValueError('WTPH2YR缺失、非正数或无穷值；不能插补权重。')
    if df[['survey_strata','survey_psu']].isna().any().any():
        raise ValueError('分层或PSU缺失。')
    groups = (df.survey_strata.astype(str) + '_' + df.survey_psu.astype(str)).to_numpy()
    excluded_cols = TARGETS | DESIGN | FASTING | REDUNDANT | {args.target, wc} | set(args.extra_drop)
    excluded_cols |= {c for c in df if c.startswith('survey_weight') or c.upper().startswith('WT')}
    # 诊断变量通常在另一个 audit 文件；即便合并回来也不得进入特征。
    excluded_cols |= {c for c in df if any(t in c.lower() for t in
                        ['hba1c', 'glucose', 'glycemic', 'insulin', 'homa', 'diq', 'lbxgh', 'lbxglu', 'diabetes'])}
    X = df.drop(columns=list(excluded_cols), errors='ignore').copy()
    cats = [c for c in X if c in CATEGORIES or not pd.api.types.is_numeric_dtype(X[c])]
    for c in cats:
        X[c] = X[c].map(lambda v: str(v) if pd.notna(v) else np.nan).astype(object)
    nums = [c for c in X if c not in cats]
    X[nums] = X[nums].replace([np.inf, -np.inf], np.nan)
    if not len(X.columns):
        raise ValueError('没有可用预测变量。')
    audit = dict(status='HBA1C_COHORT_INTERNAL_CHECKS_PASSED', quick_check_only=args.quick,
                 input_rows=input_n, exclusions_in_order=exclusions,
                 analysis_rows=len(y), positive=int(y.sum()), negative=int((y==0).sum()),
                 unweighted_prevalence=float(y.mean()), weighted_prevalence=float(np.average(y,weights=w)),
                 target=args.target, weight_column=wc, n_psu_groups=len(np.unique(groups)),
                 predictors=list(X.columns), categorical=cats,
                 excluded_columns=sorted(set(df.columns)-set(X.columns)),
                 input_sha256=source_hash,
                 provenance='User-provided merged CSV; source XPT matching not independently verified.',
                 medication_note='No reported treatment; non-diagnosed insulin NA retained as structural skip. DIQ010=2/pills NA retained with unverified DIQ160 skip when DIQ160 unavailable.',
                 oral_medication_skip_not_independently_checkable_n=unresolved_skip,
                 no_labels_or_source_rows_written=True,
                 options=vars(args), versions={m:importlib.metadata.version(m) for m in
                    ['numpy','pandas','scikit-learn','matplotlib']})
    return df, X, y, w, groups, cats, audit


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
        ('num', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True,
                                                  keep_empty_features=True)),
                          ('scale', StandardScaler())]), nums),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent',
                                                  add_indicator=True, keep_empty_features=True)),
                          ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cats)
    ], sparse_threshold=0)


def configs(model, method, quick=False):
    hyper = {'lr': [{'C': 0.1}, {'C': 1.0}],
             'rf': [{'max_depth': 6, 'min_samples_leaf': 10},
                    {'max_depth': None, 'min_samples_leaf': 20}],
             'xgb': [{'max_depth': 2, 'min_child_weight': 5},
                     {'max_depth': 3, 'min_child_weight': 10}]}[model]
    balance = {'baseline': [None], 'class_weight': [0.5, 1.0],
               'ros': [0.25, 0.5, 1.0], 'rus': [0.25, 0.5]}[method]
    if quick:
        hyper, balance = hyper[:1], balance[:1]
    return [dict(model=model, method=method, hyper=h, balance=b) for h in hyper for b in balance]


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


def balance_training(y, w, method, value, seed):
    """返回行索引与训练权重。重复抽样不产生新的独立受试者。"""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    idx = np.arange(len(y))
    fit_w = w.copy().astype(float)
    if method == 'class_weight':
        weighted_ratio = w[neg].sum() / w[pos].sum()
        fit_w[pos] *= weighted_ratio ** value
    elif method == 'ros':
        extra = max(0, int(np.ceil(value * len(neg))) - len(pos))
        idx = np.concatenate([idx, rng.choice(pos, size=extra, replace=True)])
    elif method == 'rus':
        keep = min(len(neg), int(np.ceil(len(pos) / value)))
        idx = np.concatenate([pos, rng.choice(neg, size=keep, replace=False)])
    fit_w = fit_w[idx]
    # ROS复制原权重，RUS保留原权重：有意改变训练代价，不声称还原人群。
    # 固定总权重为原始训练人数，减少样本复制对正则化总尺度的影响。
    fit_w *= len(y) / fit_w.sum()
    return idx, fit_w


def predict_fit(config, A, y, w, B, args, seed):
    idx, fit_w = balance_training(y, w, config['method'], config['balance'], seed)
    model = estimator(config, args, seed)
    model.fit(A[idx], y[idx], sample_weight=fit_w)
    return model.predict_proba(B)[:, 1]


def tune(model, method, prepared, y, w, args, seed):
    best = None
    for config in configs(model, method, args.quick):
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


def metrics(y, p, w, predicted):
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1], sample_weight=w).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    return dict(AP=float(average_precision_score(y, p, sample_weight=w)),
                ROC_AUC=float(roc_auc_score(y, p, sample_weight=w)),
                sensitivity=float(recall), specificity=float(tn / (tn + fp)),
                precision=float(precision), F1=float(2 * precision * recall / (precision + recall))
                if precision + recall else 0.0,
                Brier=float(np.average((y - p) ** 2, weights=w)),
                weighted_prevalence=float(np.average(y, weights=w)))


def plots(oof, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind in ['pr', 'calibration']:
        fig, ax = plt.subplots(figsize=(9, 6))
        for name, d in oof.groupby('combination'):
            if name == 'nested_selected':
                continue
            y, p, w = d.y.to_numpy(), d.probability.to_numpy(), d.survey_weight.to_numpy()
            if kind == 'pr':
                precision, recall, _ = precision_recall_curve(y, p, sample_weight=w)
                ax.plot(recall, precision, label=name, alpha=0.8)
            else:
                # 共同固定概率区间；每个点用调查权重计算预测与观察比例。
                bins = np.minimum((p * 10).astype(int), 9)
                points = [(np.average(p[bins == b], weights=w[bins == b]),
                           np.average(y[bins == b], weights=w[bins == b]))
                          for b in np.unique(bins)]
                a, b = zip(*points)
                ax.plot(a, b, marker='o', markersize=3, label=name, alpha=0.8)
        if kind == 'pr':
            base = oof[oof.combination == oof.combination.iloc[0]]
            ax.axhline(np.average(base.y, weights=base.survey_weight), color='gray', linestyle='--')
            ax.set(xlabel='Recall', ylabel='Precision', title='Survey-weighted PR curves (pooled outer OOF)')
            name = 'pr_curves.png'
        else:
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set(xlabel='Weighted mean predicted probability', ylabel='Weighted observed proportion',
                   title='Raw probability calibration (pooled outer OOF)')
            name = 'calibration.png'
        ax.legend(fontsize=7, loc='best')
        fig.tight_layout()
        fig.savefig(out / name, dpi=160)
        plt.close(fig)


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
    df, X, y, w, groups, cats, audit = load_data(args)
    if 'xgb' in args.models:
        audit['versions']['xgboost'] = importlib.metadata.version('xgboost')
    (out / 'data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"{audit['status']} | n={len(y)}, positive={y.sum()}, weight={audit['weight_column']}", flush=True)
    print('排除记录：' + json.dumps(audit['exclusions_in_order'], ensure_ascii=False), flush=True)
    print('输入CSV只读；当前标签与HbA1c规则一致；药物跳题及来源核验限制见data_audit.json。', flush=True)
    outer = make_splits(y, groups, args.outer_folds, args.seed)
    rows, predictions = [], []
    for fold, (tr, te) in enumerate(outer, start=1):
        print(f'外层 {fold}/{len(outer)}；训练阳性 {y[tr].sum()}，测试阳性 {y[te].sum()}', flush=True)
        Xtr = X.iloc[tr].reset_index(drop=True)
        yt, wt = y[tr], w[tr]
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
        winners = []
        for model in args.models:
            for method in METHODS:
                best = tune(model, method, prepared, yt, wt, args, args.seed + fold * 100)
                threshold = threshold_at_recall(yt, best['oof'], wt, args.recall_target)
                p = predict_fit(best['config'], A, yt, wt, B, args, args.seed + fold * 100)
                pred = (p >= threshold).astype(int)
                combination = f'{model}_{method}'
                row = dict(fold=fold, combination=combination, model=model, method=method,
                           inner_AP=best['inner_ap'], threshold=threshold, n_test=len(te),
                           positives_test=int(y[te].sum()), config=json.dumps(best['config']),
                           **metrics(y[te], p, w[te], pred))
                rows.append(row)
                frame = pd.DataFrame(dict(participant_id=df.participant_id.iloc[te].to_numpy(),
                                          fold=fold, combination=combination, y=y[te],
                                          probability=p, predicted=pred, threshold=threshold,
                                          survey_weight=w[te], psu_group=groups[te]))
                predictions.append(frame)
                winners.append((best['inner_ap'], row, frame))
                print(f"  {combination:18s} inner AP={best['inner_ap']:.4f}, outer AP={row['AP']:.4f}", flush=True)
        # 用内层分数选模型/方法，不看外层结果。复用同一拟合所得外层预测。
        _, selected_row, selected_frame = max(winners, key=lambda t: t[0])
        rows.append({**selected_row, 'combination': 'nested_selected'})
        predictions.append(selected_frame.assign(combination='nested_selected'))
        pd.DataFrame(rows).to_csv(out / 'fold_metrics.csv', index=False)
    result = pd.DataFrame(rows)
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_csv(out / 'oof_predictions.csv', index=False)
    selected = result[result.combination == 'nested_selected']
    selected.to_csv(out / 'nested_selection_metrics.csv', index=False)
    keys = ['AP', 'ROC_AUC', 'sensitivity', 'specificity', 'precision', 'F1', 'Brier']
    summary = result[result.combination != 'nested_selected'].groupby('combination')[keys].agg(['mean', 'std'])
    summary.columns = ['_'.join(c) for c in summary.columns]
    summary = summary.sort_values('AP_mean', ascending=False)
    summary.insert(0, 'descriptive_rank', np.arange(1, len(summary) + 1))
    summary.to_csv(out / 'comparison.csv')
    plots(oof, out)
    top = summary.index[0]
    explanation = (
        f"状态：{audit['status']}；quick={args.quick}\n"
        f"按外层平均调查加权 AP 的探索性第一名：{top}\n"
        f"其平均 AP：{summary.loc[top, 'AP_mean']:.4f}\n"
        f"内部选择模型+方法这一完整流程的外层平均 AP：{selected.AP.mean():.4f}\n"
        'comparison.csv 的 std 是折间标准差，不是标准误或95%置信区间。\n'
        '外层排名只作描述；挑选最高外层成绩后，不能将其宣称为无偏最终测试表现。\n'
        'nested_selection_metrics.csv 评估的是在内层选择模型+方法的过程，而非一个固定最终模型。\n'
        '请结合敏感度、精确率、特异度、Brier和校准图判断；小差异不代表显著胜出。\n'
        '阈值仅由内部OOF确定；外层敏感度不保证恰好达到设定目标。\n'
        'AP是average precision，不是梯形积分PR-AUC。\n'
        '本次仅核对合并CSV内部一致性；原始XPT来源未独立复核，用药跳题限制见data_audit.json。\n'
        'PSU分组交叉验证不等于完整调查设计推断；当前未实现设计型置信区间或外部验证。\n'
        '当前不保存最终模型，避免把未经校准和外部验证的筛查模型直接部署。\n'
    )
    (out / 'interpretation.txt').write_text(explanation, encoding='utf-8')
    print('\n' + explanation)
    unchanged = hashlib.sha256(Path(args.csv).read_bytes()).hexdigest() == audit['input_sha256']
    audit['input_unchanged_after_run'] = unchanged
    (out / 'data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'输入CSV未改变：{unchanged}')
    print(f'结果目录：{out.resolve()}')


if __name__ == '__main__':
    main()
