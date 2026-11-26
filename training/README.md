# How the ranking weights were chosen

The app scores genes with a fixed weighted sum:

```
score = 0.40·literature + 0.40·LLM relevance + 0.15·LLM evidence + 0.05·network
```

These weights were set by hand. To check whether learned weights would do better, I retrained the **pairwise learning-to-rank** method from my earlier research on the signals the live app can actually compute. I then compared it against the hand-set weights with cross-validation. Script: [`scripts/train_weights.py`](../scripts/train_weights.py). Full output: [`weights_report.json`](weights_report.json).

## Setup

| | |
|---|---|
| Data | 251 KEGG pathway cases from the frozen research feature store. These are literature-derived candidate genes with previously saved LLM evidence labels; no new model calls. |
| Signals | Paper breadth; direct-finding ratio, background ratio and evidence consensus (from LLM sentence labels); STRING degree and literature-supported STRING neighbours |
| Label | Gene is a member of the case's KEGG target pathway (a proxy for topic relevance) |
| Model | Linear pairwise logistic ranker (RankNet-style) with L2 = 0.01. Signals are z-scored within each case; at most 512 positive/negative pairs per case; every case weighted equally; fitted with L-BFGS-B. |
| Evaluation | 5-fold cross-validation split by pathway, so every pathway is scored by a model that never saw it. Each method selects its top-10 genes, which are scored with the metrics below. |

The metrics:
- **P@10:** the share of the 10 slots filled by target-pathway members.
- **R@1:** whether the target pathway ranks first in over-representation analysis (ORA).
- **MRR:** the mean reciprocal ORA rank of the target pathway.

## Results (held-out pathways)

| Ranking | P@10 | R@1 | MRR |
|---|---:|---:|---:|
| Paper count only | 0.580 | 0.725 | 0.799 |
| **Hand-set weights (used by the app)** | **0.631** | 0.777 | 0.836 |
| Learned: literature + LLM evidence | 0.606 | **0.793** | **0.841** |
| Learned: + network (all 6 signals) | **0.650** | 0.701 | 0.778 |

Paired bootstrap differences against the hand-set weights (95% CI, 5,000 resamples):

| Learned model | ΔP@10 | ΔR@1 | ΔMRR |
|---|---|---|---|
| Literature + LLM evidence | −0.024 [−0.034, −0.015] | +0.016 [−0.008, +0.044] | +0.005 [−0.013, +0.024] |
| + network | +0.020 [+0.006, +0.034] | −0.076 [−0.116, −0.036] | −0.058 [−0.087, −0.030] |

## Comparison with pathXcite GF-IDF

pathXcite ([Hamberger et al., *Briefings in Bioinformatics* 2026](https://doi.org/10.1093/bib/bbag348)) ranks literature genes with GF-IDF and has no training step. Its saved top-10 lists for the same 251 cases were re-scored with the same ORA, and the results were compared case by case with the held-out results above. Script: [`scripts/compare_with_gfidf.py`](../scripts/compare_with_gfidf.py). Output: [`gfidf_comparison.json`](gfidf_comparison.json).

| Method minus GF-IDF (95% paired bootstrap CI) | ΔP@10 | ΔR@1 | ΔMRR | R@1 wins / losses |
|---|---|---|---|---|
| GF-IDF itself | 0.574 | 0.801 | 0.851 | |
| Hand-set weights (app) | **+0.057** [+0.040, +0.073] | −0.024 [−0.060, +0.012] | −0.015 [−0.039, +0.009] | 9 / 15 |
| Learned: literature + LLM evidence | **+0.032** [+0.016, +0.048] | −0.008 [−0.044, +0.032] | −0.009 [−0.034, +0.016] | 10 / 12 |
| Learned: + network | **+0.076** [+0.057, +0.096] | −0.100 [−0.151, −0.052] | −0.072 [−0.105, −0.040] | 8 / 33 |
| Paper count only | +0.006 [−0.012, +0.024] | −0.076 [−0.128, −0.028] | −0.052 [−0.084, −0.020] | 11 / 30 |

- **Gene precision:** every literature + evidence ranker selects significantly more true pathway members than GF-IDF.
- **Pathway recovery** (pathXcite's own headline metric): the hand-set and learned literature + evidence rankers are statistically tied with GF-IDF. Neither is shown to be better or worse.
- Re-scoring reproduced GF-IDF's saved P@10 exactly, and its R@1 and MRR totals; 10 cases differed only in a lower ORA rank.
- In the separate research experiments, models trained on all 251 cases reached 214/251 significant rank-1 recoveries versus GF-IDF's 201. That result is in-sample, so it is not a held-out claim.

## Research model: held-out gene precision

The full 16-signal research ranker (from the okn-guard grouped development experiment) was also scored on its saved out-of-fold predictions: 148 pathways, each predicted by a model that never saw it.

| 148 held-out pathways | P@10 |
|---|---:|
| pathXcite GF-IDF | 0.591 |
| 16-signal research ranker | **0.751** (+0.160, 95% CI +0.134 to +0.188) |

Several of the 16 signals (KEGG co-membership, enrichment and pathway counts) come from the KEGG catalog that also supplies the labels, and the live app does not use them. Script: [`scripts/heldout_research_model_precision.py`](../scripts/heldout_research_model_precision.py). Output: [`research_model_precision.json`](research_model_precision.json).

## Interpretation

- **Adding network signals is a trade-off, not an improvement.** The learner gives STRING degree a large weight (+0.53). That picks slightly more pathway members, but pathway recovery drops sharply. Highly connected "hub" genes belong to many pathways and also strengthen competing ones. On live topics the same effect put TP53 first for *ferroptosis*. The network gain is also likely inflated, because STRING's curated channel imports KEGG pathways, the same source as the training labels.
- **Without network signals, the learned model matches the hand-set weights on pathway recovery** (the R@1 and MRR confidence intervals include 0), **but it is worse on P@10.**
- **Decision:** keep the hand-set weights. They are statistically as good as the best learned model on pathway recovery, better on gene precision, and they keep the network term small, which the results above support.

The learned weights are saved in [`learned_weights.json`](learned_weights.json) for reference. They are not used by the app.

## Limitations

- These are in-domain results. KEGG membership is a proxy for "relevant to the topic", and unannotated genes are not proven irrelevant.
- The saved evidence labels came from a different LLM and prompt than the live app's Gemini review.
- **The app's LLM relevance term (weight 0.40) has no equivalent in the saved data, so it is untested here.** Validating it would need new LLM labels for the 251 cases (about 500 calls).
- The comparison omits the app's mention-count term, which the feature store does not have.
