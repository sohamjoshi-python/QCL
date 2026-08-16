"""Interpret experiment results across all physical models.

Reads every experiment's result CSV, produces per-experiment comparison figures
(test nMSE vs. the swept variable, with error bars over seeds and the mean-guess
baseline), a headline QCL-vs-MLP figure for the disordered families, and a
statistical analysis (summary statistics + significance tests, including whether
QCL beats the mean-guess baseline and whether its error trends with dimension).

Everything is derived from the committed result CSVs, so it is cheap to re-run
and never retrains a model.

Outputs:
  results/<experiment>/images/<experiment>_test_nmse.{png,pdf}
  results/analysis/disordered_qcl_vs_mlp.{png,pdf}
  results/analysis/summary_statistics.csv
  results/analysis/significance_tests.csv
  results/analysis/report.txt   (also printed to stdout)
"""

import csv
import re
import pathlib
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as _stats
    HAVE_SCIPY = True
except Exception:  # pragma: no cover - scipy ships with pennylane, but be safe
    HAVE_SCIPY = False


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
ANALYSIS_DIR = REPO_ROOT / "results" / "analysis"

# Baseline nMSE of a model that always predicts the training mean.
MEAN_BASELINE = 1.0

# Registry of experiments: schema is either "static" (swept over n) or
# "timedep" (swept over n_layers / L).
EXPERIMENTS = [
    {"key": "xy_static", "csv": "results/xy_static/xy_experiment_results.csv",
     "schema": "static", "target": "entanglement entropy",
     "title": "Static XY"},
    {"key": "hubbard_static", "csv": "results/hubbard_static/hubbard_experiment_results.csv",
     "schema": "static", "target": "double occupancy",
     "title": "Static Hubbard"},
    {"key": "xy_disordered", "csv": "results/xy_disordered/xy_disordered_results.csv",
     "schema": "static", "target": "entanglement entropy",
     "title": "Disordered XY (d = n)"},
    {"key": "hubbard_disordered", "csv": "results/hubbard_disordered/hubbard_disordered_results.csv",
     "schema": "static", "target": "double occupancy",
     "title": "Disordered Hubbard (d = n)"},
    {"key": "xy_time_dep", "csv": "results/xy_time_dep/xy_time_dep_results.csv",
     "schema": "timedep", "target": "staggered magnetization",
     "title": "XY quench dynamics"},
    {"key": "hubbard_time_dep", "csv": "results/hubbard_time_dep/hubbard_time_dep_results.csv",
     "schema": "timedep", "target": "double occupancy",
     "title": "Hubbard quench dynamics"},
]

SWEEP_LABEL = {"static": "system size n (= input dim d)",
               "timedep": "QCL layers L"}

# Consistent styling per model series.
STYLE = {
    "QCL": {"color": "#d62728", "marker": "o"},
    "MLP": {"color": "#2ca02c", "marker": "s"},
    "Fourier full": {"color": "#1f77b4", "marker": "^"},
    "Fourier full (L=3)": {"color": "#17becf", "marker": "v"},
    "Fourier param-matched": {"color": "#9467bd", "marker": "D"},
}

_L_RE = re.compile(r"L=(\d+)")


def _to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _to_int(s):
    f = _to_float(s)
    return int(f) if f is not None else None


def load_records(exp):
    """Return a list of tidy records for one experiment.

    Each record: {model, sweep (float|None), seed (int|None),
                  train_nmse, test_nmse, n_params}.
    """
    path = REPO_ROOT / exp["csv"]
    if not path.exists():
        return None

    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            test = _to_float(row.get("test_nmse"))
            if test is None:
                continue  # skip malformed / blank lines
            train = _to_float(row.get("train_nmse"))
            n_params = _to_int(row.get("n_params"))
            raw = (row.get("model") or "").strip()

            if exp["schema"] == "static":
                sweep = _to_float(row.get("n"))
                seed = _to_int(row.get("seed"))
                model = normalize_static(raw)
            else:  # timedep
                if raw.startswith("QCL"):
                    model, sweep, seed = "QCL", _to_float(row.get("n_layers")), _to_int(row.get("seed"))
                elif "MLP" in raw:
                    model, sweep, seed = "MLP", None, _to_int(row.get("seed"))
                elif raw.startswith("Fourier full"):
                    m = _L_RE.search(raw)
                    model, sweep, seed = "Fourier full", (float(m.group(1)) if m else None), None
                elif raw.startswith("Fourier param"):
                    model, sweep, seed = "Fourier param-matched", None, None
                else:
                    continue

            records.append({
                "model": model, "sweep": sweep, "seed": seed,
                "train_nmse": train, "test_nmse": test, "n_params": n_params,
            })
    return records


def normalize_static(raw):
    if raw.startswith("QCL"):
        return "QCL"
    if "MLP" in raw:
        return "MLP"
    if raw.startswith("Fourier param"):
        return "Fourier param-matched"
    if raw.startswith("Fourier full"):
        # Keep an explicit "(L=k)" suffix so distinct full-Fourier variants
        # remain separate series.
        m = _L_RE.search(raw)
        return f"Fourier full (L={m.group(1)})" if m else "Fourier full"
    return raw


# ── Summary statistics ────────────────────────────────────────────

def summarize(records):
    """Aggregate test/train nMSE by (model, sweep). Returns list of dicts."""
    groups = defaultdict(list)
    for r in records:
        key = (r["model"], r["sweep"])
        groups[key].append(r)

    out = []
    for (model, sweep), rs in groups.items():
        tests = np.array([r["test_nmse"] for r in rs], dtype=float)
        trains = np.array([r["train_nmse"] for r in rs if r["train_nmse"] is not None],
                          dtype=float)
        nparams = next((r["n_params"] for r in rs if r["n_params"] is not None), None)
        out.append({
            "model": model,
            "sweep": "all" if sweep is None else (int(sweep) if float(sweep).is_integer() else sweep),
            "count": len(tests),
            "mean_test_nmse": float(np.mean(tests)),
            "std_test_nmse": float(np.std(tests, ddof=1)) if len(tests) > 1 else 0.0,
            "min_test_nmse": float(np.min(tests)),
            "max_test_nmse": float(np.max(tests)),
            "mean_train_nmse": float(np.mean(trains)) if len(trains) else float("nan"),
            "n_params": nparams,
        })
    # Sort by model then sweep (numeric first).
    out.sort(key=lambda d: (d["model"], (float("inf") if d["sweep"] == "all" else d["sweep"])))
    return out


# ── Significance tests ────────────────────────────────────────────

def _tests_by_sweep(records, model):
    d = defaultdict(list)
    for r in records:
        if r["model"] == model:
            d[r["sweep"]].append(r["test_nmse"])
    return {k: np.array(v, dtype=float) for k, v in d.items()}


def significance(exp, records):
    """Return a list of significance-test result rows for one experiment."""
    rows = []
    key = exp["key"]

    qcl = _tests_by_sweep(records, "QCL")
    mlp = _tests_by_sweep(records, "MLP")

    # MLP may be swept (static) or a single group (timedep, sweep=None).
    mlp_all = np.concatenate(list(mlp.values())) if mlp else np.array([])

    # 1) QCL vs MLP at each QCL sweep point (Welch's two-sample t-test).
    for sweep in sorted(qcl.keys(), key=lambda s: (s is None, s)):
        a = qcl[sweep]
        b = mlp.get(sweep) if sweep in mlp else mlp_all
        if b is None or len(a) < 2 or len(b) < 2:
            continue
        stat, p = _welch(a, b)
        rows.append({
            "experiment": key, "test": "QCL_vs_MLP (Welch t)",
            "sweep": _fmt_sweep(sweep), "mean_a": float(np.mean(a)),
            "mean_b": float(np.mean(b)), "statistic": stat, "p_value": p,
            "n_a": len(a), "n_b": len(b),
        })

    # 2) QCL vs mean-guess baseline (one-sample t-test against 1.0).
    for sweep in sorted(qcl.keys(), key=lambda s: (s is None, s)):
        a = qcl[sweep]
        if len(a) < 2:
            continue
        stat, p = _ttest_1samp(a, MEAN_BASELINE)
        rows.append({
            "experiment": key, "test": "QCL_vs_baseline1.0 (1-sample t)",
            "sweep": _fmt_sweep(sweep), "mean_a": float(np.mean(a)),
            "mean_b": MEAN_BASELINE, "statistic": stat, "p_value": p,
            "n_a": len(a), "n_b": 0,
        })

    # 3) Trend: does QCL test nMSE increase with the swept variable?
    xs, ys = [], []
    for r in records:
        if r["model"] == "QCL" and r["sweep"] is not None:
            xs.append(float(r["sweep"]))
            ys.append(float(r["test_nmse"]))
    if len(set(xs)) >= 2:
        slope, p = _linregress(np.array(xs), np.array(ys))
        rows.append({
            "experiment": key, "test": "QCL_trend_vs_sweep (linregress slope)",
            "sweep": "all", "mean_a": slope, "mean_b": 0.0,
            "statistic": slope, "p_value": p, "n_a": len(xs), "n_b": 0,
        })
    return rows


def _welch(a, b):
    if HAVE_SCIPY:
        r = _stats.ttest_ind(a, b, equal_var=False)
        return float(r.statistic), float(r.pvalue)
    return _welch_manual(a, b)


def _ttest_1samp(a, popmean):
    if HAVE_SCIPY:
        r = _stats.ttest_1samp(a, popmean)
        return float(r.statistic), float(r.pvalue)
    n = len(a)
    se = np.std(a, ddof=1) / np.sqrt(n)
    t = (np.mean(a) - popmean) / se if se > 0 else float("inf")
    return float(t), _t_sf_normal(t, n - 1)


def _linregress(x, y):
    if HAVE_SCIPY:
        r = _stats.linregress(x, y)
        return float(r.slope), float(r.pvalue)
    # Fallback: slope via least squares, p from normal approx on the t-stat.
    n = len(x)
    xm, ym = x.mean(), y.mean()
    sxx = np.sum((x - xm) ** 2)
    slope = np.sum((x - xm) * (y - ym)) / sxx
    resid = y - (ym + slope * (x - xm))
    s2 = np.sum(resid ** 2) / (n - 2) if n > 2 else float("inf")
    se = np.sqrt(s2 / sxx) if sxx > 0 else float("inf")
    t = slope / se if se > 0 else float("inf")
    return float(slope), _t_sf_normal(t, n - 2)


def _welch_manual(a, b):
    ma, mb = np.mean(a), np.mean(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    na, nb = len(a), len(b)
    se = np.sqrt(va / na + vb / nb)
    t = (ma - mb) / se if se > 0 else float("inf")
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return float(t), _t_sf_normal(t, df)


def _t_sf_normal(t, df):
    # Two-sided p-value via a normal approximation (only used if scipy absent).
    from math import erfc, sqrt
    return float(erfc(abs(t) / sqrt(2)))


def _fmt_sweep(sweep):
    if sweep is None:
        return "all"
    return str(int(sweep)) if float(sweep).is_integer() else str(sweep)


# ── Plotting ──────────────────────────────────────────────────────

def _series_points(records, model):
    """Return (sorted_xs, means, stds) for a model's numeric-sweep points."""
    by = defaultdict(list)
    for r in records:
        if r["model"] == model and r["sweep"] is not None:
            by[float(r["sweep"])].append(r["test_nmse"])
    xs = sorted(by)
    means = [float(np.mean(by[x])) for x in xs]
    stds = [float(np.std(by[x], ddof=1)) if len(by[x]) > 1 else 0.0 for x in xs]
    return xs, means, stds


def _constant_series(records, model):
    """Mean/std for a model that has no sweep (single group)."""
    vals = [r["test_nmse"] for r in records if r["model"] == model and r["sweep"] is None]
    if not vals:
        return None
    v = np.array(vals, dtype=float)
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else 0.0)


def plot_experiment(exp, records):
    models = []
    for r in records:
        if r["model"] not in models:
            models.append(r["model"])

    all_x = sorted({float(r["sweep"]) for r in records if r["sweep"] is not None})
    if not all_x:
        return None

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for model in models:
        style = STYLE.get(model, {"color": "#333333", "marker": "x"})
        xs, means, stds = _series_points(records, model)
        if xs:
            ax.errorbar(xs, means, yerr=stds, label=model, capsize=3,
                        marker=style["marker"], color=style["color"],
                        linewidth=1.5, markersize=6)
        else:
            const = _constant_series(records, model)
            if const is not None:
                mean, std = const
                ax.hlines(mean, all_x[0], all_x[-1], color=style["color"],
                          linestyle="--", linewidth=1.5,
                          label=f"{model} (mean)")
                if std > 0:
                    ax.fill_between([all_x[0], all_x[-1]], mean - std, mean + std,
                                    color=style["color"], alpha=0.15)

    ax.axhline(MEAN_BASELINE, color="gray", linestyle=":", linewidth=1.2,
               label="mean-guess baseline")

    ax.set_yscale("log")
    ax.set_xticks(all_x)
    ax.set_xlabel(SWEEP_LABEL[exp["schema"]], fontsize=12)
    ax.set_ylabel("test nMSE", fontsize=12)
    ax.set_title(f"{exp['title']}  —  target: {exp['target']}", fontsize=12)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    out_dir = REPO_ROOT / "results" / exp["key"] / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / f"{exp['key']}_test_nmse"
    fig.savefig(base.with_suffix(".png"), dpi=150)
    fig.savefig(base.with_suffix(".pdf"), dpi=300)
    plt.close(fig)
    return base.with_suffix(".png")


def plot_disordered_overview(all_records):
    """Headline figure: QCL vs MLP test nMSE vs d for both disordered families."""
    families = [("xy_disordered", "XY", "#d62728"),
                ("hubbard_disordered", "Hubbard", "#ff7f0e")]
    if not all(k in all_records for k, _, _ in families):
        return None

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for key, label, color in families:
        recs = all_records[key]
        xq, mq, sq = _series_points(recs, "QCL")
        xm, mm, sm = _series_points(recs, "MLP")
        ax.errorbar(xq, mq, yerr=sq, marker="o", color=color, linewidth=1.8,
                    capsize=3, label=f"QCL — {label}")
        ax.errorbar(xm, mm, yerr=sm, marker="s", color=color, linewidth=1.8,
                    linestyle="--", capsize=3, alpha=0.7, label=f"MLP — {label}")

    ax.axhline(MEAN_BASELINE, color="gray", linestyle=":", linewidth=1.2,
               label="mean-guess baseline")
    ax.set_yscale("log")
    ax.set_xlabel("input dimension d (= system size n)", fontsize=12)
    ax.set_ylabel("test nMSE", fontsize=12)
    ax.set_title("Disordered models: QCL degrades with d while MLP stays low",
                 fontsize=11)
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    base = ANALYSIS_DIR / "disordered_qcl_vs_mlp"
    fig.savefig(base.with_suffix(".png"), dpi=150)
    fig.savefig(base.with_suffix(".pdf"), dpi=300)
    plt.close(fig)
    return base.with_suffix(".png")


# ── Report writing ────────────────────────────────────────────────

def stars(p):
    if p is None or np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    all_summ = []
    all_sig = []
    all_records = {}
    report_lines = []
    figures = []

    def log(line=""):
        print(line)
        report_lines.append(line)

    log("=" * 74)
    log("EXPERIMENT RESULTS - INTERPRETATION")
    log("=" * 74)
    if not HAVE_SCIPY:
        log("[warning] scipy not found; p-values use a normal approximation.")

    for exp in EXPERIMENTS:
        records = load_records(exp)
        if records is None:
            log(f"\n[skip] {exp['key']}: {exp['csv']} not found.")
            continue
        if not records:
            log(f"\n[skip] {exp['key']}: no parseable rows.")
            continue

        all_records[exp["key"]] = records

        log("\n" + "-" * 74)
        log(f"{exp['title']}   ({exp['key']}, target = {exp['target']})")
        log("-" * 74)

        summ = summarize(records)
        for s in summ:
            s["experiment"] = exp["key"]
        all_summ.extend(summ)

        # Print a compact summary table.
        log(f"{'model':<26}{'sweep':>7}{'n':>4}{'test nMSE (mean+/-std)':>26}")
        for s in summ:
            log(f"{s['model']:<26}{str(s['sweep']):>7}{s['count']:>4}"
                f"{s['mean_test_nmse']:>14.4f} +/- {s['std_test_nmse']:<8.4f}")

        sig = significance(exp, records)
        all_sig.extend(sig)
        if sig:
            log("\n  significance tests:")
            for t in sig:
                log(f"    {t['test']:<38} sweep={t['sweep']:<4} "
                    f"stat={t['statistic']:>8.3f}  p={t['p_value']:.3g} {stars(t['p_value'])}")

        fig = plot_experiment(exp, records)
        if fig:
            figures.append(fig)
            log(f"\n  figure: {fig.relative_to(REPO_ROOT)}")

    overview = plot_disordered_overview(all_records)
    if overview:
        figures.append(overview)
        log(f"\nheadline figure: {overview.relative_to(REPO_ROOT)}")

    # Write CSVs.
    summ_path = ANALYSIS_DIR / "summary_statistics.csv"
    with open(summ_path, "w", newline="", encoding="utf-8") as f:
        fields = ["experiment", "model", "sweep", "count", "mean_test_nmse",
                  "std_test_nmse", "min_test_nmse", "max_test_nmse",
                  "mean_train_nmse", "n_params"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in all_summ:
            w.writerow({k: s.get(k, "") for k in fields})

    sig_path = ANALYSIS_DIR / "significance_tests.csv"
    with open(sig_path, "w", newline="", encoding="utf-8") as f:
        fields = ["experiment", "test", "sweep", "mean_a", "mean_b",
                  "statistic", "p_value", "n_a", "n_b"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in all_sig:
            w.writerow({k: t.get(k, "") for k in fields})

    log("\n" + "=" * 74)
    log("ARTIFACTS")
    log("=" * 74)
    log(f"  summary statistics : {summ_path.relative_to(REPO_ROOT)}")
    log(f"  significance tests : {sig_path.relative_to(REPO_ROOT)}")
    log(f"  figures            : {len(figures)} written")

    report_path = ANALYSIS_DIR / "report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")


if __name__ == "__main__":
    main()
