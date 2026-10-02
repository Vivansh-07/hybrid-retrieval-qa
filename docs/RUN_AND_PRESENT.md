# Run it yourself and present it

Every command here has been tested on this PC (RTX 3050 Laptop GPU, Python 3.13,
Windows 11). Run them in order the first time. Times are from that test run.
Latency (ms/query) numbers vary a little between runs, and more if something else is
using the GPU at the same time. Close other heavy programs before step 3.

---

## 0. Where to run the commands: VS Code is fine

VS Code's built-in terminal **is** a terminal, so every command below works there
exactly as in Command Prompt. For the presentation, VS Code is the better choice:
code, terminal and output files are all in one window.

1. Open VS Code → **File → Open Folder…** → `D:\Deep Learning\Claude\hybrid-retrieval-qa`
2. Use **Command Prompt** as the terminal (PowerShell blocks the activation script):
   `Ctrl+Shift+P` → **Terminal: Select Default Profile** → **Command Prompt**.
3. Open the terminal: **Terminal → New Terminal** (or press ``Ctrl+` ``).
   It opens *inside the project folder*, which is where every command must run.
4. Install the **Python** extension (Extensions icon on the left, search "Python",
   publisher Microsoft) if you have not already.
5. `Ctrl+Shift+P` → **Python: Select Interpreter** → pick
   `.venv\Scripts\python.exe` (shows as *Python 3.13 ('.venv')*).
6. In the terminal, activate the environment:
   ```
   .venv\Scripts\activate
   ```
   The prompt now starts with `(.venv)`. **Every command below assumes you see
   `(.venv)`.** Do this each time you open a new terminal. (VS Code often does it for
   you once step 5 is done.)

For the presentation, make the text readable from the back of the room:
`Ctrl+=` zooms the whole window. `Ctrl+Shift+V` on any `.md` file shows it formatted.

### Everything lives on D:, nothing on C:

| What | Where |
|---|---|
| Python packages (PyTorch etc., ~6 GB) | `.venv\` (this project's own environment) |
| SciFact + the four models (~1.6 GB) | `data\hf_cache\`, set automatically in `src/hybridqa/__init__.py` |
| pip's download cache | `data\pip-cache\`, set in `.venv\pip.ini` |
| Demo temp files | `data\gradio_tmp\` |

`.venv\` and `data\` are in `.gitignore`, so none of this goes to GitHub.
The one thing still on C: is Python 3.13 itself (`C:\Program Files\Python313`), which
was already installed.

---

## 1. One-time setup (already done on this PC; needed again only on a new PC)

Run in Command Prompt from the project folder:

| # | Command | What you should see |
|---|---|---|
| 1 | `nvidia-smi` | A table naming *NVIDIA GeForce RTX 3050* |
| 2 | `python --version` | `Python 3.13.x` |
| 3 | `python -m venv .venv` | Creates the `.venv` folder (the project's own Python) |
| 4 | `.venv\Scripts\activate` | Prompt starts with `(.venv)` |
| 5 | `mkdir data\_tmp` then `set TMP=%CD%\data\_tmp` and `set TEMP=%CD%\data\_tmp` | pip unpacks downloads on D: instead of C: (lasts until you close this terminal) |
| 6 | `python -m pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126 --no-cache-dir` | GPU build of PyTorch (~2.5 GB download) |
| 7 | `python -m pip install -r requirements.txt --no-cache-dir` | sentence-transformers, faiss, rank-bm25, datasets, gradio, … |
| 8 | `python -m pip install -e .` | Registers `src/hybridqa` so `python -m hybridqa.cli` works |
| 9 | `python -c "import torch; print(torch.cuda.is_available())"` | `True` |

Why step 6 is separate: the ordinary `pip install torch` gives the **CPU-only** build on
Windows. The `--index-url …/cu126` line fetches the build compiled for CUDA 12.6, which
your driver (CUDA 13.0) supports.

> Always type `python -m pip`, never plain `pip`, so packages go into the `.venv`
> Python that runs the code.

**Warnings you can ignore:** `huggingface_hub … symlinks` (Windows caching note) and
`Loading weights: …` progress bars.

---

## 2. The runs, in order

The first run of anything needs **internet**: it downloads SciFact (~5 MB) and the
models (~100 MB–1 GB each) once, into `data\hf_cache\`. After that it works offline.

### Step 1: Unit tests (seconds)
```
python -m pytest tests/ -q
```
**Expect:** `34 passed`.
**What it shows:** the metrics (Recall, MRR, nDCG) and RRF are implemented correctly;
the tests compare them to hand-worked answers. They also check the config overrides
the embedding-cache naming and the significance comparisons.

### Step 2: BM25 baseline (≈3 min first time, including downloads)
```
python -m hybridqa.cli ablation --variants bm25
```
**Expect:** first these lines:
```
[pipeline] loaded 5183 docs, 300 judged queries
[pipeline] chunked into 21015 passages (4.05 per doc)
[pipeline] BM25 index built in …
[pipeline] dense index built in …            (≈1–2 min: embeds all 21,015 passages)
```
then a one-row table with nDCG@10, Recall@10, Recall@100, MRR@10, latency.
Later runs say `dense index loaded from cache` instead: the embeddings are saved in
`data\indexes\`. For the paper's index-build time, quote the first ("built") run.
**Sanity check:** Recall@10 should be in the 0.6–0.8 band. If it is, data loading,
chunking and evaluation are all working.
**Log it:** add an entry to `docs/EXPERIMENT_LOG.md`.

### Step 3: The full experiment (≈13 min)
```
python scripts/run_ablation.py --sweep-depth 10 25 50 100 --sweep-fusion-k 10 30 60 100
```
**What it does:** runs all four variants on all 300 claims, then the paired bootstrap
significance test, then re-runs fusion+rerank at 4 reranking depths (accuracy vs
speed) and fusion at 4 RRF-k values (robustness).
**Expect:** the four-row ablation table, then `=== Paired bootstrap on nDCG@10 ===`
with one line per pair of variants (gain, p-value, how many questions got
better/worse/same, and "significant" or "not significant"), then one line per sweep setting.
**Files written to `results/`:**

| File | Contains |
|---|---|
| `ablation.csv` | **The main table** (open it in VS Code or Excel) |
| `metrics.json` | Same numbers plus build times, chunk count, full config |
| `significance.json` | p-values for all 6 pairs of variants: is each difference real or noise? (p < 0.05 = real) |
| `sweep_rerank_depth.json` | Accuracy and ms/query at depth 10/25/50/100 |
| `sweep_rrf_k.json` | Fusion accuracy at k = 10/30/60/100 |
| `runs/*.json` | Per-question rankings (used by the failure analysis) |

To recompute `significance.json` from the saved run files **without** re-running the
13-minute experiment (seconds):
```
python scripts/significance.py
```
A p-value printed as `< 0.0001` means none of the 10,000 bootstrap resamples produced a gain
that large by chance. Write it that way in the paper, never as "p = 0".

### Step 3b: One-change experiments (≈5 min each)
Change **one** setting for one run with `--set key=value`; no YAML editing needed.
Results go to their own folder, `results\set_<key>=<value>\`, so they never overwrite
the main `results\` files from step 3. The full config used is saved in that folder's
`metrics.json`.
```
python scripts/run_ablation.py --set chunking.strategy=none
python scripts/run_ablation.py --variants fusion --set fusion.method=nsf
```
The first is the **chunking study** (whole abstracts vs 3-sentence windows); the second
is **rank fusion vs score fusion**. A mistyped key stops immediately with a list of the
valid keys. Log each run.

Then compare the run with the main results, method by method (seconds):
```
python scripts/compare_runs.py "results/set_chunking.strategy=none"
python scripts/compare_runs.py "results/set_fusion.method=nsf"
```
For each method it prints nDCG@10 in the main run and in this run, the difference, a
p-value and how many questions got better/worse/same, and saves `comparison.json` in
that folder.

> **Recall@100 caveat.** With 3-sentence windows, the 100 retrieved *passages* belong to
> only about 47-58 different *documents* per question. So "Recall@100" in the main table
> means "recall within the top-100 passages", and it cannot be compared with a
> whole-abstract run (exactly 100 documents). nDCG@10, Recall@10 and MRR@10 are
> unaffected: compare configurations on those.

### Step 3c: Stronger models (done on 30 Sep; results in `results\dense_bge_small*`)
Checks whether the findings hold with better models. MiniLM stays the baseline; these are
reported next to it, never instead of it. Models download into `data\hf_cache\` on D:.
```
python scripts/run_ablation.py --set dense.model=BAAI/bge-small-en-v1.5 --set "dense.query_prefix=Represent this sentence for searching relevant passages: " --out results/dense_bge_small
python scripts/compare_runs.py results/dense_bge_small
```
(≈9 min; 129 MB download.) And the stronger reranker on top (≈8 min; 1.1 GB download):
```
python scripts/run_ablation.py --variants fusion+rerank --set dense.model=BAAI/bge-small-en-v1.5 --set "dense.query_prefix=Represent this sentence for searching relevant passages: " --set rerank.model=BAAI/bge-reranker-base --out results/dense_bge_small__rerank_bge_base
python scripts/compare_runs.py results/dense_bge_small__rerank_bge_base --baseline results/dense_bge_small
```
Keep the quotes around the `dense.query_prefix=...` part: the prefix ends with a space.

What they showed (nDCG@10): dense 0.679 → **0.723** with BGE (significant); fusion then
equals dense alone (0.721); neither reranker improves on that (0.712 and 0.714).

### Step 3d: Fine-tune the dense encoder (done on 30 Sep; ≈10 min training + ≈4 min test)
This is the part of the project where a model is actually **trained**. It uses SciFact's
809 *training* claims (none of them is a test claim): 689 to train, 120 held out as a dev
set that picks the best epoch. The test claims are only used afterwards.
```
python scripts/finetune_dense.py --epochs 12
python scripts/run_ablation.py --set dense.model=data/models/all-MiniLM-L6-v2-scifact --out results/dense_minilm_finetuned
python scripts/compare_runs.py results/dense_minilm_finetuned
```
**Expect (first command):** one line per epoch with the training loss going down and
`dev nDCG@10` going up (0.691 before training → about 0.770), then the saved model path.
The model is written to `data\models\all-MiniLM-L6-v2-scifact\` (88 MB, on D:, not on
GitHub; re-run the first command to recreate it). The history goes to
`results\finetune_all-MiniLM-L6-v2-scifact\history.json`, and `make_plots.py` turns it
into `paper\figures\finetune_curve_all-MiniLM-L6-v2-scifact.png`.

What it showed on the test claims (nDCG@10): dense 0.679 → **0.742** (p < 0.0001), the
best result in the project; fusion 0.710 → 0.731 (p = 0.013).

Be able to explain `scripts/finetune_dense.py`: `contrastive_loss` (why the target for row
i is column i), `best_chunk_pairs` (where the positives come from) and the dev-set early
stopping.

### Step 4: Figures (seconds)
```
python scripts/make_plots.py
```
**Writes:** `paper/figures/ablation_ndcg.png`, `recall_curve.png`, `latency_accuracy.png`.
Click them in the VS Code file explorer to view. These go into the paper and slides.

- `ablation_ndcg.png` and `recall_curve.png` are drawn 3.5 inches wide, exactly one IEEE
  column, so the text prints at full size. `latency_accuracy.png` is two panels spanning
  both columns (`figure*` in `main.tex`).
- Lines differ by dash pattern and marker shape, not only colour, so they still read on a
  black-and-white printout.
- In `latency_accuracy.png`, "none" means fusion without the reranker, and the dotted lines
  mark BM25 and dense alone. Read it as "does reranking more candidates buy accuracy, and
  at what cost?"

### Step 5: Failure-analysis worksheet (seconds)
```
python scripts/failure_analysis.py --top-n 40
```
**Writes:** `results/failure_analysis.csv`: the 40 questions the system gets most wrong.
**It is already generated for you** (committed on 1 Oct), so you only label it. Each row
shows the claim, the gold abstract, the rank every variant gave it and the system's top
three; see "How to label the failure worksheet" in §4 for the six categories. The script
refuses to overwrite a worksheet that already has labels. This manual part is what the
examiner will probe hardest. Do not automate it.

### Step 6: Cited answers (≈2 min for 50 questions)
```
python -m hybridqa.cli answer --limit 50
```
**Writes:** `results/answers.json`: the question, the answer, the five passages it was
given, which ones it cited, and citation precision/recall against the gold documents.
**Known issue, and a finding for the paper:** the default reader `flan-t5-base` replies
with only a citation like `[3].` for 49 of the 50 claims. See §5 below.

The answers of both readers (generative and extractive) were generated on 1 Oct, and the
blind scoring sheet is ready: `results/answer_scoring.csv`. Score it with
`docs/ANSWER_RUBRIC.md`, commit after every 25 answers, then:
```
python scripts/answer_sheet.py --summarize
```

### Step 7: One question from the command line (≈1 min: rebuilds BM25 and loads the models each time)
```
python -m hybridqa.cli query "Does vitamin D reduce respiratory infections?" --answer
```

### Step 8: The web demo
```
python scripts/demo.py
```
Takes ≈1.5 min to start: it builds the BM25 index, then prints
`[demo] loading the models with one warm-up question...` and answers one question by
itself so every model is already on the GPU. Then it prints
`Running on local URL: http://127.0.0.1:7860` and opens that page in your browser.
Leave the terminal open. Closing it or pressing `Ctrl+C` stops the demo.

Because of that warm-up, **the first question you ask is already fast** (≈4–5 s
including the answer; measured 2 Oct). Start the demo a few minutes before you present.

**For the presentation, run it with your fine-tuned encoder** (the best retriever):
```
python scripts/demo.py --set dense.model=data/models/all-MiniLM-L6-v2-scifact
```
(The folder `data\modelsll-MiniLM-L6-v2-scifact` must exist; if it doesn't, run
`python scripts/finetune_dense.py --epochs 12` first.)

To run the demo with the stronger dense model instead (the header shows which models are active):
```
python scripts/demo.py --set dense.model=BAAI/bge-small-en-v1.5 --set "dense.query_prefix=Represent this sentence for searching relevant passages: "
```

Or with your fine-tuned encoder (the best-performing setup):
```
python scripts/demo.py --set dense.model=data/models/all-MiniLM-L6-v2-scifact
```

- **Live demo tab:** pick a SciFact claim from the dropdown (or type any question) →
  **Retrieve**. Four columns show what BM25, dense, fusion and fusion+rerank each
  return; ✅ marks the correct paper. Below: the generated answer with citations.
- **Results tab:** your `ablation.csv` table and the three figures.

---

## 3. Presentation day

**At home, the day before (needs internet the first time):**
1. Run steps 1–6 above so `results/` and `paper/figures/` are filled with your numbers.
2. Try the demo and pick **3 claims** to show: one where BM25 wins, one where dense
   wins, one where reranking fixes the order. Write them down. (For example, claim 660,
   *"Ivermectin is used to treat onchocerciasis."*: BM25 ranks the gold paper first but
   the cross-encoder pushes it out of the top 5, a live `reranker_error` example.
   `results/failure_analysis.csv` lists more candidates.)
3. Take screenshots of the ablation table, the figures and the demo, as a backup in
   case anything fails in class.

**In class (about 10 minutes):**

| # | Show | Say |
|---|---|---|
| 1 | VS Code: `src/hybridqa/pipeline.py`, the `retrieve()` function | "All four strategies share one index; only this function changes. That makes it a controlled ablation." |
| 2 | Run `python -m pytest tests/ -q` live | "The metrics are implemented from their definitions and checked against hand-computed values." |
| 3 | Start the demo (command above, with the fine-tuned encoder) a few minutes before your slot | ≈1.5 min to load; the first question is then fast. Talk through the pipeline diagram in the README meanwhile |
| 4 | `results/ablation.csv` | The main result: which strategy wins and at what cost |
| 5 | `results/significance.json` | Which improvements are statistically real (p < 0.05) and which are not |
| 6 | `paper/figures/latency_accuracy.png` | "Reranking depth is the accuracy/speed dial" |
| 7 | Browser: the demo, with your 3 prepared claims | Point out the ✅ moving between columns |
| 8 | `results/failure_analysis.csv` | Your hand labels: *why* the system still fails |

Don't run `run_ablation.py` live. It takes several minutes and doesn't need to be
shown running; the log and `results/` files are the evidence.

---

## 4. Daily GitHub activity

Your repo: <https://github.com/Vivansh-07/hybrid-retrieval-qa> (private for now).

**Make sure the green squares actually appear:**
1. GitHub → **Settings → Emails**: `vivanshgarg05@gmail.com` must be listed and verified.
   Commits are credited to you by that email.
2. Your profile page → **Contribution settings → Private contributions** ✔.
   Otherwise commits to a private repo do not show on your graph.
3. Commit to the `main` branch. Commits on other branches don't count until merged.

**Daily routine (2 minutes, VS Code terminal):**
```
git status
git add .
git commit -m "Run dense baseline; log results"
git push
```
Or use VS Code's **Source Control** panel (the branch icon on the left): type the
message, click **Commit**, then **Sync Changes**.

**Several commits a day is fine, as long as each one is a real step.** Commit every time
you finish a meaningful piece: a batch of labels, a rewritten section, a set of slides.
The examiner may look at `git log`, and many small *real* commits are strong evidence of
your own work. Empty or trivial commits made only for green squares are easy to spot,
so don't make them.

### Done so far
- 25 Sep – 1 Oct: pipeline, fixes, D:-only setup, BM25 + full ablation (your run),
  significance for all pairs, figures, chunking study, score fusion, stronger encoder and
  reranker, **fine-tuning (0.742)**, query-subset analysis, paper drafts of Method, Setup,
  Results and Limitations, problem statement, failure worksheet.

### Day-by-day plan to the deadline (about 10 Oct), 2–3 commits a day

Commit after **each** row item, not once per day.

| Date | Work, one commit each | Commit message idea |
|---|---|---|
| Fri 2 Oct | ~~Re-time the main ablation~~ (done 1 Oct)<br>(1) Label failure claims 1–10<br>(2) Label failure claims 11–20<br>(3) Decide the answer reader (§5): keep flan-t5-base and report it, or try flan-t5-large before scoring | `Failure analysis: claims 1-10 labelled`<br>`Failure analysis: claims 11-20 labelled`<br>`Choose the answer reader` |
| Sat 3 Oct | (1) Label claims 21–30<br>(2) Label claims 31–40, run `--summarize`<br>(3) Write the Failure Analysis section of the paper | `Failure analysis: claims 21-30`<br>`Failure analysis: all 40 labelled`<br>`Paper: failure analysis section` |
| Sun 4 Oct | ~~Rubric, 50 answers per reader, scoring sheet~~ (done 1 Oct)<br>(1) Read `docs/ANSWER_RUBRIC.md` until you can apply it without looking<br>(2) Score answers 1–50 of the sheet | `Answer scoring: 50/100` |
| Mon 5 Oct | (1) Score answers 51–100, run `python scripts/answer_sheet.py --summarize`<br>(2) Answer-quality subsection in the paper | `Answer scoring: 100/100`<br>`Paper: answer quality` |
| Tue 6 Oct | (1) Rewrite Method in your own words<br>(2) Rewrite Experimental Setup in your own words | `Paper: method in my words`<br>`Paper: setup in my words` |
| Wed 7 Oct | (1) Rewrite Results in your own words<br>(2) Write the Introduction<br>(3) Conclusion and the abstract's numbers | `Paper: results in my words`<br>`Paper: introduction`<br>`Paper: abstract and conclusion` |
| Thu 8 Oct | (1) Slides 1–6<br>(2) Slides 7–12 | `Slides: problem and method`<br>`Slides: results and conclusions` |
| Fri 9 Oct | (1) Fresh-clone test: clone into a new folder **on D:**, follow this file from step 1, fix what breaks<br>(2) Viva rehearsal with `docs/VIVA_PREP.md`; write down what you stumbled on | `Reproducibility pass`<br>`Viva prep notes` |
| Sat 10 Oct | (1) Final read-through; export the paper PDF from Overleaf into `paper/`<br>(2) Buffer | `Final paper PDF` |

If a day slips, drop items in the order listed in `CLAUDE.md` §7 ("If behind, cut in
this order"), never the four-row ablation, the failure analysis or the significance test.

### How to label the failure worksheet

Open `results\failure_analysis.csv` in Excel. For each of the 40 rows, read the claim,
the gold abstract's title and opening, and the ranks, then type **one** category, the
main cause, into `category` and one sentence of why into `notes`. Save as CSV.

| Category | Choose it when | Columns that help |
|---|---|---|
| `terminology_mismatch` | the gold abstract says the same thing in different words | low `word_overlap`; BM25 ranks the gold far down |
| `ambiguous_query` | the claim is so general that many abstracts fit it | the top-3 abstracts all look plausible |
| `missing_evidence` | the gold abstract does not actually state what the claim says | read `gold_abstract_start` |
| `conflicting_evidence` | the top abstracts discuss the same question with opposite findings | read `top3_retrieved` |
| `granularity` | the gold abstract is found but just outside the top 10, because a different passage of it was matched | gold rank 11–20 for most variants |
| `reranker_error` | fusion had the gold abstract high, the reranker pushed it down | `reranker_regression` = 1; compare `gold_rank_fusion` with `gold_rank_fusion_rerank` |

Then:
```
python scripts/failure_analysis.py --summarize
```
It checks that every label is one of the six, prints the distribution for the paper,
and writes `results\failure_summary.json`. Do the labelling yourself: this is the part
of the project the examiner will probe hardest.

After grading, make the repo public for your resume:
```
gh repo edit Vivansh-07/hybrid-retrieval-qa --visibility public --accept-visibility-change-consequences
```

---

## 5. Open decision: the answer generator

`flan-t5-base` (250M parameters) tends to output just `[1]` rather than an answer
sentence, and does not reliably say "Not enough evidence" when the passages do
not answer. Options:

1. **Keep it and report it.** It is a genuine, explainable finding (small instruction-tuned
   models copy the citation format instead of answering). Put it in Limitations.
2. **Switch to extractive mode** (`answer.mode: extractive` in `configs/scifact.yaml`):
   `roberta-base-squad2` selects an exact span from the top passage, so the citation
   is correct by construction. It worked well in testing.
3. **Try `google/flan-t5-large`** (780M): better at following the instruction, ~3 GB
   download, tight but should fit in 4 GB VRAM.

Try an option without editing the YAML: `--set` writes to its own folder, so your
default answers.json is kept:
```
python -m hybridqa.cli answer --limit 50 --set answer.mode=extractive
python -m hybridqa.cli answer --limit 50 --set answer.generative_model=google/flan-t5-large
```
Whichever you pick, change one thing, run it, and log it.
