# Gene Prioritizer

**Enter a biomedical topic and a number _k_. Get the top-_k_ human genes, ranked from PubMed literature with LLM evidence review and a citation for every claim.**

🔗 **Live demo:** https://ramya2003-cyber.github.io/gene-prioritization/ &nbsp;·&nbsp; **API docs:** https://gene-prioritizer-api.onrender.com/docs

![Screenshot](docs/screenshot.png)

## Try it
1. Open the live demo, type a topic (e.g. `ferroptosis`, `Alzheimer's disease`, `hypoxia`) and choose _k_ (1–50).
2. Watch the live progress. Results show each gene's LLM relevance label, a one-line rationale with PubMed links, the exact evidence sentences, a score breakdown and an interaction network. They can be exported as CSV or JSON.
3. The three example chips load instantly. A new topic takes about 1–2 minutes. The free server sleeps when idle, so the first search after a break needs about a minute more.

## Where the LLM is used
Text mining finds *which* genes a set of abstracts mentions. It cannot tell whether a mention is a real finding about the topic or an incidental one. That judgement is the LLM's job:

- For each shortlisted gene, the model (Gemini, through an OpenAI-compatible API) receives the topic and up to three **exact sentences** in which the gene appears.
- It returns strict JSON:
  - a **relevance label** for the gene: core, supporting, peripheral or unrelated
  - an **evidence label** for each sentence: direct finding, indirect, or background
  - a short **rationale** that cites the sentences. Citations are converted into PubMed IDs, so every claim can be checked.
- These labels make up 55% of the final score.

**Example (ferroptosis):** ranked by paper count alone, the top 10 includes a mis-tagged abbreviation, ATL1 (#6), and the general signalling genes AKT1 (#8) and TP53 (#10). The LLM labelled ATL1 *unrelated* and AKT1 and TP53 *supporting*. They fell to #20, #14 and #11, making room for ATF4 and DHODH, which it labelled *core*. The "Other reviewed candidates" table on the results page shows every demoted gene with the model's reason.

### Guardrails
- The model judges **only the sentences it is given**, with temperature 0 and JSON output.
- **Every item is validated.** Unknown genes, made-up passage IDs and invalid labels are dropped and listed in the run notes.
- **Failures degrade gracefully.** A failed or rate-limited batch affects only its own genes, which fall back to literature signals, and the page says so.
- **No gene is invented.** If fewer than _k_ human genes are found, the shortfall is reported.
- **Usage is bounded:** at most 4 LLM calls per search, a per-IP hourly limit, a 24 h result cache and a cap on concurrent searches.
- **Provider-agnostic:** any OpenAI-compatible endpoint works by changing environment variables, including a local model (see below).

## How it works

```mermaid
flowchart LR
    A[Topic + k] --> B[Europe PMC<br/>250 most relevant abstracts]
    B --> C[PubTator3<br/>gene mentions + sentences]
    C --> D[MyGene.info<br/>verify human genes]
    D --> E[Shortlist by<br/>paper breadth]
    E --> F[STRING<br/>interactions]
    E --> G[LLM review<br/>label each sentence + gene]
    F --> H[Weighted score]
    G --> H
    H --> I[Top-k genes with citations]
```

```
score = 0.40·literature + 0.40·LLM relevance + 0.15·LLM evidence + 0.05·network
```

Each term is scaled from 0 to 1 before weighting:

| Term | Weight | What it measures | How it is calculated |
|---|---:|---|---|
| **Literature** | 0.40 | How widely the topic's literature mentions the gene | log(1 + papers) + 0.25·log(1 + mentions), divided by the highest value among the topic's genes. *Papers* is the number of the 250 abstracts that mention the gene; *mentions* is the total number of times it is tagged in them. |
| **LLM relevance** | 0.40 | The LLM's overall judgement of the gene's role in the topic | core = 1, supporting = 0.6, peripheral = 0.25, unrelated = 0 (0 if not reviewed) |
| **LLM evidence** | 0.15 | How strong the individual sentences are | The average of the LLM's labels for up to 3 sentences per gene: direct finding = 1, indirect = 0.5, background = 0.1 |
| **Network** | 0.05 | How connected the gene is to the other candidates | The number of high-confidence STRING interactions (score ≥ 0.7) with other shortlisted genes, divided by the highest such count |

**Example (ferroptosis):** GPX4 is mentioned in 65 of the 250 abstracts (the most of any gene), is labelled *core*, has 3 *direct* sentences and has 8 interactions. That gives 0.40 + 0.40 + 0.15 + 0.036 = **0.99**. TFRC is in 13 abstracts (literature 0.26), *core* (0.40), and has 2 direct and 1 background sentence (0.105) and 6 interactions (0.027), for **0.79**.

The network term is deliberately small, so highly connected "hub" genes such as TP53 or AKT1 cannot outrank genes the literature is actually about.

### How the weights were chosen
The weights are set by hand. I tested them against **learned weights from a pairwise learning-to-rank model**, trained on 251 KEGG pathway cases and evaluated with 5-fold cross-validation by pathway (details in [`training/`](training/README.md)):

| Ranking (held-out pathways) | Gene precision@10 | Pathway R@1 | MRR |
|---|---:|---:|---:|
| pathXcite GF-IDF (published method, no training) | 0.574 | **0.801** | **0.851** |
| Paper count only | 0.580 | 0.725 | 0.799 |
| **Hand-set weights (used)** | **0.631** | 0.777 | 0.836 |
| Learned: literature + evidence | 0.606 | 0.793 | 0.841 |
| Learned: + network signals | 0.650 | 0.701 | 0.778 |

- **Network signals:** the learner gave them a large weight, which selected slightly more pathway members but cut pathway recovery by 7.6 points (statistically significant).
- **Without network signals:** the learned model tied the hand-set weights on pathway recovery but was worse on precision.
- **Against pathXcite** ([Hamberger et al., *Briefings in Bioinformatics* 2026](https://doi.org/10.1093/bib/bbag348)), paired on the same 251 pathways: the hand-set weights put significantly more true pathway genes in the top 10 (+5.7 points, 95% CI +4.0 to +7.3). Pathway recovery is statistically tied (R@1 −2.4 points, CI −6.0 to +1.2). See [`training/gfidf_comparison.json`](training/gfidf_comparison.json).
- **Result:** the hand-set weights were kept. The LLM relevance term has no saved equivalent, so it is not covered by this test.

## Run with your own local LLM
The LLM step uses the standard OpenAI chat-completions API, so a self-hosted model can replace Gemini without code changes. Nothing is sent to an external LLM; only the public literature APIs (Europe PMC, PubTator3, MyGene, STRING) are called.

1. **Serve your model** with an OpenAI-compatible server, for example:
   ```bash
   vllm serve <model-path-or-name> --port 8001          # vLLM  -> http://localhost:8001/v1
   ollama serve && ollama pull <model>                  # Ollama -> http://localhost:11434/v1
   ```
2. **Point the backend at it.** Create `.env` in the repo root (copy `.env.example`):
   ```bash
   LLM_BASE_URL=http://localhost:8001/v1   # your server's /v1 address
   LLM_MODEL=<model name as the server reports it>
   LLM_API_KEY=local                       # any non-empty text; local servers ignore it
   LLM_TIMEOUT=300                         # seconds per request; raise for slower hardware
   ```
3. **Start the app:**
   ```bash
   pip install -r backend/requirements.txt
   cd backend && uvicorn app.main:app --port 8000      # API
   cd frontend && python -m http.server 5500           # UI at http://127.0.0.1:5500
   ```
4. **Check it:** open http://127.0.0.1:8000/api/health. It should show `"llm_enabled": true` and your model name. Results pages show the model that reviewed the evidence.

**What the model must do:** each request is about 3–4k tokens (12 genes, up to 3 sentences each), and the model must answer with one JSON object. The app requests JSON mode, which vLLM, Ollama and llama.cpp support. If a server ignores it, JSON is still extracted from plain text. Tool calling is not used. Invalid or failed answers never break a search; they are dropped and reported.

**Tested:** `qwen2.5:7b` on Ollama, on a laptop RTX 4050 GPU, with the settings above. A ferroptosis search ran end to end in about 200 s, with 2 LLM calls reviewing 20 genes and no invalid output. The labels were sensible (GPX4 and SLC7A11 *core*, the hub gene AKT1 *peripheral*), but weaker than a larger model's: rationales usually omitted sentence citations, and some well-known genes, such as TFRC, were under-rated. A ~27B model should do noticeably better.

## Tech stack

```
frontend/   HTML/CSS/JavaScript, no build step       → GitHub Pages
            live progress (Server-Sent Events), SVG network graph, CSV/JSON export
backend/    Python 3.11, FastAPI                      → Render
  app/literature.py  Europe PMC + PubTator3 + MyGene
  app/llm.py         prompt, batched calls, JSON validation
  app/ranking.py     weighted scoring
  app/pipeline.py    orchestration, caching, progress events
  app/main.py        REST + streaming endpoints, rate limiting
  tests/             offline tests, all external APIs mocked
training/   weight-validation study (learning-to-rank vs hand-set weights)
```

**Run locally:** `pip install -r backend/requirements.txt`, put a free Gemini key in `.env` as `LLM_API_KEY` (see `.env.example`), then run `uvicorn app.main:app` from `backend/`, and `python -m http.server 5500` from `frontend/`. Tests: `python -m pytest -q` in `backend/`.

## Limitations
- Only the 250 most relevant abstracts are read, so rare or recently reported genes can be missed.
- LLM labels are model judgements, not expert curation. Scores rank literature evidence and are **not** probabilities of causality.
- For research exploration only, not clinical use.

## Background
This app grew out of my research on literature-based gene prioritization. That work compared GF-IDF, learning-to-rank and evolution-strategy rankers on KEGG pathway-recovery benchmarks. The earlier rankers over-rewarded network hub genes on live topics: for example, TP53 ranked #1 for *ferroptosis*, ahead of GPX4. This version adds the LLM evidence layer and caps the network term to fix that.

## License
All rights reserved. This code is shared for review as part of my M.Tech project; please contact me before reusing it.
