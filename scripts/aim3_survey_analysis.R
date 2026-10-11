# =====================================================================
# BMI5109 – Aim 3：哪些人群最容易患有未诊断糖尿病（R 版）
#
# 变量：年龄、性别、医疗保险、家庭收入贫困比
# 方法：
#   第 1 步  调查设计（全样本建设计，再取子总体分析）
#   第 2 步  各组加权未诊断率 + 95% CI
#   第 3 步  Rao-Scott 卡方检验 + 趋势检验（年龄、收入）
#   第 4 步  加权多因素 logistic 回归（模型 1：年龄+性别；模型 2：+医保+收入）
#   第 5 步  敏感性分析：删除收入未报告者，重做模型 2
#   第 6 步  补充分析：在所有糖尿病患者中，未诊断所占比例
#
# ---------------- 怎么运行 ----------------
# 保持仓库 scripts/、data/ 和 results/ 目录结构。
# 相对路径以仓库根目录为准，不依赖 R 的工作目录。
# 默认读取 data/…_verified.csv；结果写到 results/aim3/ 的新建运行子目录。
# 缺少 survey 包会自动安装。
#
# 运行方式任选一种：
#   R 自带界面（RGui）：菜单「文件」→「运行 R 脚本文件…」→ 选中本文件
#   RStudio：打开本文件，点右上角 Source
#   命令行：Rscript scripts/aim3_survey_analysis.R
# 也可以手动指定数据：Rscript aim3_survey_analysis.R 路径/数据.csv
#
# 输出：
#   Table_Aim3.csv                     报告用汇总表（比例、p 值、aOR）
#   prevalence_by_group.csv            第 2 步
#   rao_scott_tests.csv                第 3 步
#   regression_models.csv              第 4、5 步
#   supplementary_among_diabetes.csv   第 6 步
# =====================================================================

DATA_FILE <- "data/nhanes_diabetes_2021_2023_verified.csv"
OUTPUT_FOLDER <- "results/aim3"

# ---- 缺少 survey 包时自动安装（不会弹出选镜像的窗口）----
if (!requireNamespace("survey", quietly = TRUE)) {
  install.packages("survey", repos = "https://cloud.r-project.org")
}
suppressPackageStartupMessages(library(survey))
options(survey.lonely.psu = "adjust")

# ---- 自动找到本代码文件所在的文件夹 ----
script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f) == 1) return(dirname(normalizePath(f)))               # Rscript 运行
  for (i in rev(seq_len(sys.nframe()))) {                               # source() / 菜单运行
    of <- sys.frame(i)$ofile
    if (!is.null(of)) return(dirname(normalizePath(of)))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- rstudioapi::getSourceEditorContext()$path                      # RStudio
    if (nzchar(p)) return(dirname(normalizePath(p)))
  }
  getwd()
}
HERE <- script_dir()
REPO_ROOT <- normalizePath(file.path(HERE, ".."), winslash = "/", mustWork = TRUE)
resolve_path <- function(path) {
  path <- chartr("\\", "/", path.expand(path))
  if (!grepl("^([A-Za-z]:/|/)", path)) path <- file.path(REPO_ROOT, path)
  path
}

args <- commandArgs(trailingOnly = TRUE)
DATA <- resolve_path(if (length(args) >= 1) args[1] else DATA_FILE)
if (!file.exists(DATA)) stop("找不到分析数据文件: ", DATA)
cat("读取数据:", normalizePath(DATA), "\n")
OUT_BASE <- resolve_path(OUTPUT_FOLDER)
dir.create(OUT_BASE, recursive = TRUE, showWarnings = FALSE)
OUT <- tempfile(pattern = paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_"),
                tmpdir = OUT_BASE)
if (!dir.create(OUT)) stop("无法创建结果目录: ", OUT)

d <- read.csv(DATA, stringsAsFactors = FALSE)

# ---------------------------------------------------------------------
# 0. 权重：HbA1c 是血液指标，应使用抽血权重 WTPH2YR
# ---------------------------------------------------------------------
if ("WTPH2YR" %in% names(d)) {
  d$w <- d$WTPH2YR
  cat("使用权重: WTPH2YR\n")
} else {
  d$w <- d$survey_weight_mec_2yr
  warning("数据中没有 WTPH2YR，暂用 MEC 权重（survey_weight_mec_2yr）。结果仅为初步结果，补上 WTPH2YR 后需重新运行。")
}
d <- d[!is.na(d$w) & d$w > 0, ]

# ---------------------------------------------------------------------
# 1. 变量分组
# ---------------------------------------------------------------------
d$undx <- d$diabetes_undiagnosed                       # 结局：1 = 未诊断糖尿病

d$age3 <- cut(d$age_years, breaks = c(17, 39, 59, Inf),
              labels = c("18-39", "40-59", ">=60"))     # 参照组：18-39

d$sex2 <- factor(d$sex, levels = c("female", "male"),
                 labels = c("Female", "Male"))           # 参照组：Female

d$insured <- factor(d$has_health_insurance, levels = c(1, 0),
                    labels = c("Insured", "Uninsured"))  # 参照组：Insured；缺失保持 NA

r <- d$family_income_poverty_ratio
d$income5 <- ifelse(is.na(r), "Not reported",
             ifelse(r < 1, "<1",
             ifelse(r < 2, "1-1.99",
             ifelse(r < 4, "2-3.99", ">=4"))))
d$income5 <- factor(d$income5,
                    levels = c(">=4", "2-3.99", "1-1.99", "<1", "Not reported"))  # 参照组：>=4
d$income4 <- droplevels(factor(ifelse(d$income5 == "Not reported", NA,
                                      as.character(d$income5)),
                               levels = c(">=4", "2-3.99", "1-1.99", "<1")))

# 趋势检验用的数值评分（只对有顺序的变量）
d$age_score    <- as.numeric(d$age3)                                   # 1,2,3 = 越来越老
d$income_score <- c(4, 3, 2, 1)[as.numeric(d$income4)]                 # 1 = 最低收入 ... 4 = 最高

# 分析人群（与机器学习部分统一）：
#   排除 已诊断或正在用药者、用药情况不明者（降糖药问题拒答/不知道，编码 7/9）、孕妇
med_unknown <- if ("diq070_taking_diabetic_pills" %in% names(d)) {
  d$diabetes_diagnosed == 0 & d$diq070_taking_diabetic_pills %in% c(7, 9)
} else {
  stop("数据中缺少 diq070_taking_diabetic_pills 列，请使用 nhanes_diabetes_2021_2023_verified.csv")
}
d$in_aim3 <- d$diabetes_diagnosed == 0 & !med_unknown & !(d$pregnant %in% 1)
# 补充分析人群：所有糖尿病患者（已诊断 + 未诊断）、非孕妇
d$in_dm   <- d$diabetes == 1 & !(d$pregnant %in% 1)

# ---------------------------------------------------------------------
# 第 1 步：调查设计 —— 先用全样本建设计，再用 subset() 取子总体
#          （不能先删人再建设计，否则标准误会算错）
# ---------------------------------------------------------------------
des  <- svydesign(ids = ~survey_psu, strata = ~survey_strata,
                  weights = ~w, nest = TRUE, data = d)
sub  <- subset(des, in_aim3)
cat(sprintf("Aim 3 分析样本: %d 人, 未诊断糖尿病 %d 例\n",
            sum(d$in_aim3), sum(d$undx[d$in_aim3])))

vars <- c(Age = "age3", Sex = "sex2", `Health insurance` = "insured", Income = "income5")

# ---------------------------------------------------------------------
# 第 2 步：各组加权未诊断率 + 95% CI
# ---------------------------------------------------------------------
prev_table <- function(design, data_rows) {
  out <- list()
  tot <- svyciprop(~undx, design, method = "logit")
  ci  <- attr(tot, "ci")
  out[[1]] <- data.frame(Characteristic = "Overall", Category = "All",
                         n = nrow(data_rows), cases = sum(data_rows$undx),
                         pct = 100 * coef(tot), lcl = 100 * ci[1], ucl = 100 * ci[2])
  for (lab in names(vars)) {
    v <- vars[[lab]]
    for (lv in levels(data_rows[[v]])) {
      ds  <- subset(design, get(v) == lv)
      rows <- data_rows[!is.na(data_rows[[v]]) & data_rows[[v]] == lv, ]
      if (nrow(rows) == 0) next
      est <- svyciprop(~undx, ds, method = "logit")
      ci  <- attr(est, "ci")
      out[[length(out) + 1]] <- data.frame(
        Characteristic = lab, Category = lv, n = nrow(rows), cases = sum(rows$undx),
        pct = 100 * coef(est), lcl = 100 * ci[1], ucl = 100 * ci[2])
    }
  }
  res <- do.call(rbind, out)
  res$`Weighted % (95% CI)` <- sprintf("%.1f (%.1f-%.1f)", res$pct, res$lcl, res$ucl)
  rownames(res) <- NULL
  res
}
prev <- prev_table(sub, d[d$in_aim3, ])
write.csv(prev, file.path(OUT, "prevalence_by_group.csv"), row.names = FALSE)

# ---------------------------------------------------------------------
# 第 3 步：Rao-Scott 卡方检验 + 趋势检验
#   收入的卡方和趋势只用 4 个有收入的组（“未报告”不是一个收入水平）
# ---------------------------------------------------------------------
rs <- function(v, design) {
  ds <- subset(design, !is.na(get(v)))
  t  <- svychisq(as.formula(paste("~undx +", v)), ds, statistic = "F")  # Rao-Scott 二阶校正
  t$p.value
}
trend <- function(score, design) {
  ds <- subset(design, !is.na(get(score)))
  m  <- svyglm(as.formula(paste("undx ~", score)), design = ds, family = quasibinomial())
  coef(summary(m))[score, "Pr(>|t|)"]
}
tests <- data.frame(
  Characteristic = names(vars),
  p_RaoScott = c(rs("age3", sub), rs("sex2", sub), rs("insured", sub), rs("income4", sub)),
  p_trend    = c(trend("age_score", sub), NA, NA, trend("income_score", sub))
)
write.csv(tests, file.path(OUT, "rao_scott_tests.csv"), row.names = FALSE)

# ---------------------------------------------------------------------
# 第 4 步：加权多因素 logistic 回归
#   两个模型用同一批人（医保无缺失者），便于比较
# ---------------------------------------------------------------------
sub_m <- subset(sub, !is.na(insured))
m1 <- svyglm(undx ~ age3 + sex2, design = sub_m, family = quasibinomial())
m2 <- svyglm(undx ~ age3 + sex2 + insured + income5, design = sub_m, family = quasibinomial())

# 第 5 步：敏感性分析 —— 删除收入未报告者
sub_cc <- subset(sub_m, income5 != "Not reported")
m2_cc  <- svyglm(undx ~ age3 + sex2 + insured + income4, design = sub_cc, family = quasibinomial())

or_table <- function(m, name) {
  est <- coef(m)[-1]
  ci  <- confint(m)[-1, , drop = FALSE]
  p   <- coef(summary(m))[-1, "Pr(>|t|)"]
  data.frame(Model = name, Term = names(est),
             aOR = exp(est), lcl = exp(ci[, 1]), ucl = exp(ci[, 2]), p = p,
             `aOR (95% CI)` = sprintf("%.2f (%.2f-%.2f)", exp(est), exp(ci[, 1]), exp(ci[, 2])),
             check.names = FALSE, row.names = NULL)
}
models <- rbind(or_table(m1, "Model 1: age + sex"),
                or_table(m2, "Model 2: + insurance + income"),
                or_table(m2_cc, "Sensitivity: Model 2, complete-case income"))
write.csv(models, file.path(OUT, "regression_models.csv"), row.names = FALSE)

# ---------------------------------------------------------------------
# 第 6 步：补充分析 —— 所有糖尿病患者中，未诊断的比例
#   回答“得了糖尿病后，谁更容易没被医疗系统发现”
# ---------------------------------------------------------------------
dm <- subset(des, in_dm)
supp <- prev_table(dm, d[d$in_dm, ])
supp$p_RaoScott <- NA
for (i in seq_along(vars)) {
  v <- if (vars[[i]] == "income5") "income4" else vars[[i]]
  supp$p_RaoScott[supp$Characteristic == names(vars)[i]][1] <- rs(v, dm)
}
write.csv(supp, file.path(OUT, "supplementary_among_diabetes.csv"), row.names = FALSE)

# ---------------------------------------------------------------------
# 汇总成报告用的表：每组的比例、p 值、模型 2 的 aOR
# ---------------------------------------------------------------------
term_of <- function(var, lv) paste0(var, lv)
tab <- prev[, c("Characteristic", "Category", "n", "cases", "Weighted % (95% CI)")]
tab$`p (Rao-Scott)` <- ""
tab$`p for trend`   <- ""
tab$`aOR (95% CI), Model 2` <- ""
fmtp <- function(p) ifelse(is.na(p), "", ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))
m2t <- models[models$Model == "Model 2: + insurance + income", ]
for (i in seq_along(vars)) {
  lab <- names(vars)[i]; v <- vars[[i]]
  rows <- which(tab$Characteristic == lab)
  tab$`p (Rao-Scott)`[rows[1]] <- fmtp(tests$p_RaoScott[i])
  tab$`p for trend`[rows[1]]   <- fmtp(tests$p_trend[i])
  for (r_ in rows) {
    term <- term_of(v, tab$Category[r_])
    tab$`aOR (95% CI), Model 2`[r_] <- if (term %in% m2t$Term)
      m2t$`aOR (95% CI)`[m2t$Term == term] else "1.00 (reference)"
  }
}
tab$`aOR (95% CI), Model 2`[tab$Characteristic == "Overall"] <- ""
write.csv(tab, file.path(OUT, "Table_Aim3.csv"), row.names = FALSE)

cat("\n===== 各组加权未诊断率与检验 =====\n")
print(tab, row.names = FALSE)
cat("\n===== 回归模型（aOR）=====\n")
print(models[, c("Model", "Term", "aOR (95% CI)", "p")], row.names = FALSE, digits = 3)
cat("\n===== 补充分析：糖尿病患者中未诊断的比例 =====\n")
print(supp[, c("Characteristic", "Category", "n", "cases", "Weighted % (95% CI)", "p_RaoScott")],
      row.names = FALSE, digits = 3)
cat(sprintf("\n全部结果已保存到文件夹: %s\n", normalizePath(OUT)))
