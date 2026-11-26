"""Held-out comparison against pathXcite GF-IDF (Hamberger et al., Briefings in Bioinformatics 2026).

Same cross-validation as train_weights.py, plus GF-IDF's saved top-10 lists re-scored with the same ORA.
Output: training/gfidf_comparison.json. Original docstring follows.

Learn the app's signal weights with pairwise learning-to-rank, and compare to alternatives.

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


    # GF-IDF (PathXcite) top-10 lists saved by the research comparison, re-scored with the same ORA.
    gf = {}
    with (report_dir / "per_pathway.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["method"] == "GF-IDF":
                gf[row["pathway_id"]] = (row["selected_genes"].split("|") if row["selected_genes"] else [], int(row["rank"] or 0), float(row["precision_at10"]))
    per_case["GF-IDF (PathXcite)"] = {}
    mism = 0
    for pid in cases:
        genes, saved_rank, saved_p = gf[pid]
        labels = dict(zip(data[pid]["symbols"], data[pid]["labels"]))
        rank = ora.outcome(pid, genes)["rank"]
        mism += (rank != saved_rank)
        per_case["GF-IDF (PathXcite)"][pid] = {"p10": sum(labels.get(g, 0) for g in genes) / 10, "r1": float(rank == 1), "rr": 1 / rank if rank else 0.0, "saved_p10": saved_p}
    print("GF-IDF rank mismatches vs saved:", mism, " p10 mismatches:", sum(abs(v["p10"]-v["saved_p10"])>1e-9 for v in per_case["GF-IDF (PathXcite)"].values()))
    rng2 = np.random.default_rng(7)
    base = per_case["GF-IDF (PathXcite)"]
    out = {}
    for m in per_case:
        row = {}
        for metric in ("p10", "r1", "rr"):
            a = np.array([per_case[m][p][metric] for p in cases]); b = np.array([base[p][metric] for p in cases])
            d = a - b
            boots = [d[rng2.integers(0, len(d), len(d))].mean() for _ in range(5000)]
            row[metric] = [round(float(a.mean()), 4), round(float(d.mean()), 4), round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)]
        if m != "GF-IDF (PathXcite)":
            a = np.array([per_case[m][p]["r1"] for p in cases]); b = np.array([base[p]["r1"] for p in cases])
            row["r1_wins_losses"] = [int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())]
        out[m] = row
        print(m, row)
    json.dump({"cases": len(cases), "folds": FOLDS, "seed": SEED, "results": out, "model_calls": 0}, open(ROOT / "training" / "gfidf_comparison.json", "w", encoding="utf-8"), indent=1)

if __name__ == "__main__":
    main()
