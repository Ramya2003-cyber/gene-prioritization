"""Learn the app's signal weights with pairwise learning-to-rank, and compare to alternatives.

Reuses the frozen research feature store (251 KEGG pathway cases, saved evidence labels,
no new LLM calls) from the okn-guard research repo. Only signals that the live app can
also compute are used, so the learned weights transfer to the app.

Method: L2-regularised linear pairwise logistic ranker (RankNet-style), within-case
z-scored signals, label = gene is a member of the case's KEGG target pathway,
<=512 sampled positive/negative pairs per case, each case weighted equally, L-BFGS-B.

Evaluation: 5-fold cross-validation split by pathway (each pathway is scored by a model
that never saw it). Metrics on the top-10 selected genes:
  P@10  share of the 10 slots filled by target-pathway members
  R@1   target pathway ranked first by over-representation analysis (ORA)
  MRR   mean reciprocal ORA rank of the target pathway

Usage (inside the okn-guard environment):
    python scripts/train_weights.py --research-repo ../okn-guard
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LIVE_FEATURES = [
    "log_paper_count",
    "direct_finding_ratio",
    "background_noise_ratio",
    "evidence_consensus_score",
    "string_degree_all",
    "string_supported_neighbors",
]
NO_NETWORK = LIVE_FEATURES[:4]
L2, FOLDS, SEED = 0.01, 5, 20261126


def zscore(x):
    std = x.std(axis=0)
    return (x - x.mean(axis=0)) / np.where(std > 0, std, 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--research-repo", default=str(ROOT.parent / "okn-guard"))
    args = ap.parse_args()
    research = Path(args.research_repo).resolve()
    sys.path[:0] = [str(research / "src"), str(research / "scripts")]
    from okn_guard.services import gp_bench as gb
    from okn_guard.services.gene_signal_policy import fit_weights, make_pairs
    from compare_full_training_gene_methods import Ora

    report_dir = research / "reports/full_training_gene_comparison_20261118"
    raw = json.loads((report_dir / "features.json").read_text(encoding="utf-8"))
    names = {}
    with (report_dir / "per_pathway.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            names[row["pathway_id"]] = row["target_pathway"]
    kegg = gb.load_gmt()
    cases = sorted(p for p in raw if p in names)
    ora = Ora(names)

    data = {}
    for pid in cases:
        symbols = sorted(raw[pid])
        full = np.array([[raw[pid][s][f] for f in LIVE_FEATURES] for s in symbols], float)
        members = kegg[names[pid]]
        data[pid] = {
            "symbols": symbols,
            "raw": full,
            "labels": np.array([s in members for s in symbols], int),
        }

    def matrix(pid, feats):
        idx = [LIVE_FEATURES.index(f) for f in feats]
        return zscore(data[pid]["raw"][:, idx])

    def learn(train, feats):
        mats = {p: matrix(p, feats) for p in train}
        w, info = fit_weights(make_pairs(mats, {p: data[p]["labels"] for p in train}, train), L2)
        return w, info

    def top10(pid, scores):
        syms = data[pid]["symbols"]
        return [syms[i] for i in sorted(range(len(syms)), key=lambda i: (-scores[i], syms[i]))[:10]]

    def evaluate(pid, genes):
        labels = dict(zip(data[pid]["symbols"], data[pid]["labels"]))
        rank = ora.outcome(pid, genes)["rank"]
        return {"p10": sum(labels[g] for g in genes) / 10, "r1": float(rank == 1), "rr": 1 / rank if rank else 0.0}

    # Baselines that need no training.
    def papers_only(pid):
        return data[pid]["raw"][:, 0]

    def app_handset(pid):
        """The app's previous hand-set weights, minus the LLM relevance term (not in the store)."""
        x = data[pid]["raw"]
        papers, direct, background = x[:, 0], x[:, 1], x[:, 2]
        indirect = np.clip(1 - direct - background, 0, 1)
        evidence = direct * 1.0 + indirect * 0.5 + background * 0.1
        degree = np.expm1(x[:, 4])
        return 0.40 * papers / max(papers.max(), 1e-9) + 0.15 * evidence + 0.05 * degree / max(degree.max(), 1)

    rng = np.random.default_rng(SEED)
    order = rng.permutation(cases)
    folds = [sorted(order[i::FOLDS]) for i in range(FOLDS)]
    methods = {
        "Paper count only": lambda pid, w: papers_only(pid),
        "Previous hand-set weights": lambda pid, w: app_handset(pid),
        "Learned pairwise, literature + LLM evidence": ("no_net", NO_NETWORK),
        "Learned pairwise, + network (all 6 signals)": ("all", LIVE_FEATURES),
    }
    per_case = {m: {} for m in methods}
    fold_weights = {"no_net": [], "all": []}
    for k, test in enumerate(folds):
        train = [p for p in cases if p not in set(test)]
        fitted = {}
        for key, feats in (("no_net", NO_NETWORK), ("all", LIVE_FEATURES)):
            fitted[key] = learn(train, feats)[0]
            fold_weights[key].append(fitted[key].tolist())
        for pid in test:
            for m, spec in methods.items():
                if callable(spec):
                    scores = spec(pid, None)
                else:
                    key, feats = spec
                    scores = matrix(pid, feats) @ fitted[key]
                per_case[m][pid] = evaluate(pid, top10(pid, scores))
        print(f"fold {k + 1}/{FOLDS} done", flush=True)

    def summary(rows):
        vals = list(rows.values())
        return {k: round(float(np.mean([v[k] for v in vals])), 4) for k in ("p10", "r1", "rr")}

    results = {m: summary(per_case[m]) for m in methods}

    # Paired bootstrap over cases: each learned model minus the previous hand-set weights.
    boot = np.random.default_rng(SEED)
    idx = [boot.integers(0, len(cases), len(cases)) for _ in range(5000)]
    diffs = {}
    for m in ("Learned pairwise, literature + LLM evidence", "Learned pairwise, + network (all 6 signals)"):
        a, b = per_case[m], per_case["Previous hand-set weights"]
        diffs[m] = {}
        for metric in ("p10", "r1", "rr"):
            d = np.array([a[p][metric] - b[p][metric] for p in cases])
            means = [d[i].mean() for i in idx]
            diffs[m][metric] = {"mean": round(float(d.mean()), 4),
                                "ci95": [round(float(np.percentile(means, 2.5)), 4),
                                         round(float(np.percentile(means, 97.5)), 4)]}

    # Final weights for the app: refit on all 251 cases.
    final = {}
    for key, feats in (("no_net", NO_NETWORK), ("all", LIVE_FEATURES)):
        w, info = learn(cases, feats)
        final[key] = {"features": feats, "weights": dict(zip(feats, [round(float(v), 4) for v in w])), "fit": {
            k: info[k] for k in ("loss", "iterations", "converged", "training_queries", "pairs")}}
    stability = {key: {"min": np.round(np.min(v, axis=0), 4).tolist(), "max": np.round(np.max(v, axis=0), 4).tolist()}
                 for key, v in fold_weights.items()}

    out = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "L2-regularised linear pairwise logistic ranker (L2=0.01, <=512 pairs/case, L-BFGS-B)",
        "labels": "KEGG_2021_Human target-pathway membership (proxy for topic relevance)",
        "cases": len(cases),
        "evaluation": f"{FOLDS}-fold cross-validation split by pathway, seed {SEED}; top-10 genes; ORA rank by raw p",
        "cv_results": results,
        "learned_minus_handset": diffs,
        "final_weights": final,
        "cv_weight_range": stability,
        "model_calls": 0,
    }
    target = ROOT / "training" / "weights_report.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    (ROOT / "training" / "learned_weights.json").write_text(
        json.dumps({"source": "scripts/train_weights.py", **final["all"]}, indent=2), encoding="utf-8")
    print(json.dumps({"cv_results": results, "learned_minus_handset": diffs, "final_weights": final,
                      "cv_weight_range": stability}, indent=2))


if __name__ == "__main__":
    main()
