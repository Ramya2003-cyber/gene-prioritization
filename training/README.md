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
