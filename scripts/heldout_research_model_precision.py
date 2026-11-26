"""Held-out gene precision@10 of the 16-signal research ranker vs pathXcite GF-IDF.

Uses the out-of-fold predictions saved by the okn-guard grouped development experiment
(reports/linear_gene_policy_20261118, 148 pathways, five outer folds) and GF-IDF's top-10
lists on the same pathways. No training and no model calls.
Note: several of the 16 signals derive from the KEGG catalog, the same source as the labels.

Usage (inside the okn-guard environment):
    python scripts/heldout_research_model_precision.py --research-repo ../okn-guard
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(); ap.add_argument("--research-repo", default=str(ROOT.parent / "okn-guard"))
research = Path(ap.parse_args().research_repo).resolve()
sys.path.insert(0, str(research / "src"))
from okn_guard.services import gp_bench as gb

kegg = gb.load_gmt()
oof = json.loads((research / "reports/linear_gene_policy_20261118/oof_predictions.json").read_text(encoding="utf-8"))
names, gf = {}, {}
with (research / "reports/full_training_gene_comparison_20261118/per_pathway.csv").open(encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        names[row["pathway_id"]] = row["target_pathway"]
        if row["method"] == "GF-IDF":
            gf[row["pathway_id"]] = row["selected_genes"].split("|") if row["selected_genes"] else []
ml, g = [], []
for pid, v in sorted(oof.items()):
    members = kegg[names[pid]]
    top = [e["gene"] for e in sorted(v["entries"], key=lambda e: e["rank"])[:10]]
    ml.append(sum(x in members for x in top) / 10)
    g.append(sum(x in members for x in gf[pid]) / 10)
ml, g = np.array(ml), np.array(g); d = ml - g
rng = np.random.default_rng(7)
boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
out = {"cases": len(d), "research_model_p10": round(float(ml.mean()), 4), "gfidf_p10": round(float(g.mean()), 4),
       "difference": round(float(d.mean()), 4),
       "ci95": [round(float(np.percentile(boots, 2.5)), 4), round(float(np.percentile(boots, 97.5)), 4)],
       "note": "Held-out (out-of-fold). Several of the 16 signals (KEGG co-membership, enrichment, pathway counts) derive from the KEGG catalog used for labels.", "model_calls": 0}
(ROOT / "training" / "research_model_precision.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(out)
