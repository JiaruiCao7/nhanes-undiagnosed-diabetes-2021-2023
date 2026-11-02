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
args <- commandArgs(trailingOnly = TRUE)
RES <- if (length(args) >= 1) args[1] else "results/aim2"
RES <- chartr("\\", "/", path.expand(RES))
if (!grepl("^([A-Za-z]:/|/)", RES)) RES <- file.path(REPO_ROOT, RES)
if (!file.exists(file.path(RES, "prevalence_by_group.csv")))
  stop("Prevalence results not found; first run scripts/aim2_survey_analysis.R.")
OUT <- file.path(RES, "Figure_Aim2.png")

prev <- read.csv(file.path(RES, "prevalence_by_group.csv"), check.names = FALSE)
reg  <- read.csv(file.path(RES, "regression_model.csv"),    check.names = FALSE)

BLUE <- "#2a6fdb"; INK <- "#1f2328"; MUTED <- "#59636e"; GRID <- "#d0d7de"
nice <- c(">=60" = "≥ 60", ">=2" = "≥ 2", "<2" = "< 2",
          "18-39" = "18–39", "40-59" = "40–59")
lab <- function(x) ifelse(x %in% names(nice), nice[x], x)

groups <- list(Age = c("18-39", "40-59", ">=60"),
               Sex = c("Female", "Male"),
               `Health insurance` = c("Insured", "Uninsured"),
               `Income-to-poverty ratio` = c(">=2", "<2", "Not reported"))
src <- c(Age = "Age", Sex = "Sex", `Health insurance` = "Health insurance",
         `Income-to-poverty ratio` = "Income")
rowsA <- data.frame(); y <- 0
for (g in names(groups)) {
  rowsA <- rbind(rowsA, data.frame(y = y, label = g, head = TRUE, pct = NA, lcl = NA, ucl = NA))
  y <- y + 1
  for (k in groups[[g]]) {
    r <- prev[prev$Characteristic == src[g] & prev$Category == k, ]
    rowsA <- rbind(rowsA, data.frame(y = y, label = lab(k), head = FALSE,
                                     pct = r$pct, lcl = r$lcl, ucl = r$ucl))
    y <- y + 1
  }
  y <- y + 0.4
}
overall <- prev$pct[prev$Characteristic == "Overall"]

terms <- c("age340-59"           = "Age 40–59 vs 18–39",
           "age3>=60"            = "Age ≥ 60 vs 18–39",
           "sex2Male"            = "Male vs female",
           "insuredUninsured"    = "Uninsured vs insured",
           "income3<2"           = "Income ratio < 2 vs ≥ 2",
           "income3Not reported" = "Income not reported vs ≥ 2")
rowsB <- reg[match(names(terms), reg$Term), ]

png(OUT, width = 11, height = 5.2, units = "in", res = 300, bg = "white")
layout(matrix(1:2, 1), widths = c(1, 1.4))


par(mar = c(4.5, 10.5, 3, 1), family = "sans", col.axis = MUTED, fg = GRID)
pts <- rowsA[!rowsA$head, ]
xmax <- ceiling(max(pts$ucl))
plot(NA, xlim = c(0, xmax), ylim = c(max(rowsA$y) + 0.6, -1.2), axes = FALSE,
     xlab = "", ylab = "", xaxs = "i")
abline(v = 0:xmax, col = GRID, lwd = 0.6)
abline(v = overall, col = MUTED, lty = 3)
text(overall, -1.2, sprintf("Overall %.1f%%", overall), col = MUTED, cex = 0.75, pos = 4)
segments(pts$lcl, pts$y, pts$ucl, pts$y, col = BLUE, lwd = 2.5)
points(pts$pct, pts$y, pch = 21, bg = BLUE, col = "white", cex = 1.4, lwd = 1.5)
axis(1, at = 0:xmax, col = GRID, col.ticks = GRID, cex.axis = 0.85)
mtext("Weighted % with undiagnosed diabetes (95% CI)", side = 1, line = 2.8, col = INK, cex = 0.85)
mtext(rowsA$label, side = 2, at = rowsA$y, las = 1, line = 0.5, adj = 1, col = INK,
      font = ifelse(rowsA$head, 2, 1), cex = 0.85)
mtext("A   Prevalence by group", side = 3, line = 1.2, adj = 0, font = 2, col = INK)


par(mar = c(4.5, 12.5, 3, 7))
plot(NA, xlim = c(0.3, 12), ylim = c(nrow(rowsB) + 0.4, 0.4), log = "x", axes = FALSE,
     xlab = "", ylab = "")
abline(v = c(0.5, 1, 2, 5, 10), col = GRID, lwd = 0.6)
abline(v = 1, col = MUTED, lty = 2)
yy <- seq_len(nrow(rowsB))
segments(rowsB$lcl, yy, rowsB$ucl, yy, col = BLUE, lwd = 2.5)
points(rowsB$aOR, yy, pch = 21, bg = BLUE, col = "white", cex = 1.4, lwd = 1.5)
axis(1, at = c(0.5, 1, 2, 5, 10), labels = c("0.5", "1", "2", "5", "10"),
     col = GRID, col.ticks = GRID, cex.axis = 0.85)
mtext("Adjusted odds ratio (95% CI), log scale", side = 1, line = 2.8, col = INK, cex = 0.85)
mtext(terms, side = 2, at = yy, las = 1, line = 0.5, col = INK, cex = 0.85)
mtext(sprintf("%.2f (%.2f–%.2f)", rowsB$aOR, rowsB$lcl, rowsB$ucl),
      side = 4, at = yy, las = 1, line = 0.5, col = MUTED, cex = 0.75)
mtext("B   Adjusted odds ratios", side = 3, line = 1.2, adj = 0, font = 2, col = INK)

invisible(dev.off())
cat("已保存:", normalizePath(OUT), "\n")
