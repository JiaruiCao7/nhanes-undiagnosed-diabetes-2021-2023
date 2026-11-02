# Aim 2: survey-weighted prevalence and adjusted associations.
# Statistical calculations retained from the teammate's aim3_simplified.R.
# Missing insurance remains NA; regression excludes these records.
# The summary p column contains Rao-Scott tests, not regression coefficient tests.
if (!requireNamespace("survey", quietly = TRUE))
  install.packages("survey", repos = "https://cloud.r-project.org")
library(survey)
options(survey.lonely.psu = "adjust")

script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f) == 1) return(dirname(normalizePath(f)))
  for (i in rev(seq_len(sys.nframe()))) {
    of <- sys.frame(i)$ofile
    if (!is.null(of)) return(dirname(normalizePath(of)))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- rstudioapi::getSourceEditorContext()$path
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
MEMBERSHIP <- resolve_path(if (length(args) >= 1) args[1] else "results/aim1/cohort_membership.csv")
DATA <- resolve_path(if (length(args) >= 2) args[2] else "data/nhanes_diabetes_2021_2023_verified.csv")
OUT_BASE <- resolve_path(if (length(args) >= 3) args[3] else "results/aim2/runs")
if (!file.exists(DATA) || !file.exists(MEMBERSHIP)) stop("Analysis data or cohort manifest not found.")
d <- read.csv(DATA)
m <- read.csv(MEMBERSHIP)
if (!all(c("participant_id", "included") %in% names(m))) stop("Invalid cohort manifest.")
if (anyNA(d$participant_id) || anyDuplicated(d$participant_id) ||
    anyNA(m$participant_id) || anyDuplicated(m$participant_id) ||
    !setequal(d$participant_id, m$participant_id)) stop("Data/manifest participant IDs differ.")
if (anyNA(m$included) || !all(m$included %in% c(0, 1))) stop("Invalid inclusion flags.")
if (!"WTPH2YR" %in% names(d) || anyNA(d$WTPH2YR) ||
    any(!is.finite(d$WTPH2YR)) || any(d$WTPH2YR <= 0)) stop("Invalid WTPH2YR weights.")
dir.create(OUT_BASE, recursive = TRUE, showWarnings = FALSE)
OUT <- tempfile(pattern = paste0(format(Sys.time(), "%Y%m%d_%H%M%S"), "_"), tmpdir = OUT_BASE)
if (!dir.create(OUT)) stop("Cannot create results directory.")

d$in_cohort <- d$participant_id %in% m$participant_id[m$included == 1]
cat("分析人群：", sum(d$in_cohort), "人，未诊断糖尿病", sum(d$diabetes_undiagnosed[d$in_cohort]),
    "例（应为 5026 人、126 例）\n")


d$y <- d$diabetes_undiagnosed
d$age3 <- cut(d$age_years, c(-Inf, 39, 59, Inf), labels = c("18-39", "40-59", ">=60"))
d$sex2 <- factor(d$sex, levels = c("female", "male"), labels = c("Female", "Male"))
d$insured <- factor(ifelse(d$has_health_insurance == 1, "Insured", "Uninsured"),
                    levels = c("Insured", "Uninsured"))           
r <- d$family_income_poverty_ratio
d$income3 <- factor(ifelse(is.na(r), "Not reported", ifelse(r >= 2, ">=2", "<2")),
                    levels = c(">=2", "<2", "Not reported")) 


des <- svydesign(ids = ~survey_psu, strata = ~survey_strata, weights = ~WTPH2YR,
                 nest = TRUE, data = d)
sub <- subset(des, in_cohort)

prev_row <- function(design, label, cat) {
  ci <- svyciprop(~y, design, method = "logit")
  data.frame(Characteristic = label, Category = cat,
             n = nrow(design$variables), cases = sum(design$variables$y),
             pct = 100 * as.numeric(ci), lcl = 100 * attr(ci, "ci")[1], ucl = 100 * attr(ci, "ci")[2])
}
vars <- list(Age = "age3", Sex = "sex2", `Health insurance` = "insured", Income = "income3")
prev <- prev_row(sub, "Overall", "All")
for (v in names(vars)) {
  x <- vars[[v]]
  for (lv in levels(d[[x]])) {
    s <- subset(sub, get(x) == lv)
    prev <- rbind(prev, prev_row(s, v, lv))
  }
}
prev$`Weighted % (95% CI)` <- sprintf("%.1f (%.1f-%.1f)", prev$pct, prev$lcl, prev$ucl)


rs <- data.frame()
for (v in names(vars)) {
  x <- vars[[v]]
  s <- subset(sub, !is.na(get(x)) & get(x) != "Not reported")
  s$variables[[x]] <- droplevels(s$variables[[x]])               
  t <- svychisq(as.formula(paste("~y +", x)), s, statistic = "F")
  rs <- rbind(rs, data.frame(Characteristic = v, p_RaoScott = unname(t$p.value)))
}


reg_des <- subset(sub, !is.na(insured))
fit <- svyglm(y ~ age3 + sex2 + insured + income3, design = reg_des, family = quasibinomial())
est <- coef(fit)[-1]; ci <- confint(fit)[-1, , drop = FALSE]
reg <- data.frame(Term = names(est), aOR = exp(est), lcl = exp(ci[, 1]), ucl = exp(ci[, 2]),
                  p = summary(fit)$coefficients[-1, 4], row.names = NULL)
reg$`aOR (95% CI)` <- sprintf("%.2f (%.2f-%.2f)", reg$aOR, reg$lcl, reg$ucl)
cat("回归人数：", nrow(reg_des$variables), "人，", sum(reg_des$variables$y), "例\n")


term_of <- function(ch, ca) {
  pre <- c(Age = "age3", Sex = "sex2", `Health insurance` = "insured", Income = "income3")[ch]
  paste0(pre, ca)
}
tab <- prev[, c("Characteristic", "Category", "n", "cases", "Weighted % (95% CI)")]
tab$p <- ""
for (i in seq_len(nrow(rs))) {
  j <- which(tab$Characteristic == rs$Characteristic[i])[1]
  tab$p[j] <- ifelse(rs$p_RaoScott[i] < 0.001, "<0.001", sprintf("%.3f", rs$p_RaoScott[i]))
}
tab$`aOR (95% CI)` <- ""
for (i in seq_len(nrow(tab))) {
  if (tab$Characteristic[i] == "Overall") next
  t <- term_of(tab$Characteristic[i], tab$Category[i])
  if (t %in% reg$Term) tab$`aOR (95% CI)`[i] <- reg$`aOR (95% CI)`[reg$Term == t]
  else if (tab$Category[i] %in% c("18-39", "Female", "Insured", ">=2")) tab$`aOR (95% CI)`[i] <- "1.00 (reference)"
}

write.csv(prev, file.path(OUT, "prevalence_by_group.csv"), row.names = FALSE)
write.csv(rs,   file.path(OUT, "rao_scott_tests.csv"),     row.names = FALSE)
write.csv(reg,  file.path(OUT, "regression_model.csv"),    row.names = FALSE)
write.csv(tab,  file.path(OUT, "Table_Aim2_final.csv"),    row.names = FALSE)
print(tab, row.names = FALSE)
cat("\n已保存到：", OUT, "\n")
