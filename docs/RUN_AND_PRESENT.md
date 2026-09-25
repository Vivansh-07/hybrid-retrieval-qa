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
**Expect:** `18 passed`.
**What it shows:** the metrics (Recall, MRR, nDCG) and RRF are implemented correctly;
the tests compare them to hand-worked answers.

### Step 2: BM25 baseline (≈3 min first time, including downloads)
```
python -m hybridqa.cli ablation --variants bm25
```
**Expect:** first these lines:
```
[pipeline] loaded 5183 docs, 300 judged queries
[pipeline] chunked into 21015 passages (4.05 per doc)
[pipeline] BM25 index built in …
[pipeline] dense index built in …
```
then a one-row table with nDCG@10, Recall@10, Recall@100, MRR@10, latency.
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
with a `delta` and `p` for each comparison, then one line per sweep setting.
**Files written to `results/`:**

| File | Contains |
|---|---|
| `ablation.csv` | **The main table** (open it in VS Code or Excel) |
| `metrics.json` | Same numbers plus build times, chunk count, full config |
| `significance.json` | p-values: is each improvement real or noise? (p < 0.05 = real) |
| `sweep_rerank_depth.json` | Accuracy and ms/query at depth 10/25/50/100 |
| `sweep_rrf_k.json` | Fusion accuracy at k = 10/30/60/100 |
| `runs/*.json` | Per-question rankings (used by the failure analysis) |

### Step 4: Figures (seconds)
```
python scripts/make_plots.py
```
**Writes:** `paper/figures/ablation_ndcg.png`, `recall_curve.png`, `latency_accuracy.png`.
Click them in the VS Code file explorer to view. These go into the paper and slides.

### Step 5: Failure-analysis worksheet (seconds)
```
python scripts/failure_analysis.py --top-n 40
```
**Writes:** `results/failure_analysis.csv`: the 40 questions the system gets most wrong.
Open it in Excel and fill in the category column **by hand** (the six categories are
listed at the top of `scripts/failure_analysis.py`). This manual part is what the
examiner will probe hardest. Do not automate it.

### Step 6: Cited answers (≈2 min for 50 questions)
```
python -m hybridqa.cli answer --limit 50
```
**Writes:** `results/answers.json`: the question, the answer, which passages it cited,
and citation precision/recall against the gold documents.
**Known issue, and a finding for the paper:** the default reader `flan-t5-base` often
replies with only a citation like `[1]` instead of a sentence. See §5 below.

### Step 7: One question from the command line (≈1 min: rebuilds BM25 and loads the models each time)
```
python -m hybridqa.cli query "Does vitamin D reduce respiratory infections?" --answer
```

### Step 8: The web demo
```
python scripts/demo.py
```
Takes ≈50 s to start (it builds the BM25 index), prints
`Running on local URL: http://127.0.0.1:7860` and opens that page in your browser.
Leave the terminal open. Closing it or pressing `Ctrl+C` stops the demo.

**The first question after starting is slow (≈40 s)** while the models load onto the
GPU; after that each question takes under a second. So always ask one warm-up
question before you present.

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
| 3 | Start `python scripts/demo.py` in the terminal | ≈50 s to load, then ask one warm-up question (≈40 s). Talk through the pipeline diagram in the README meanwhile |
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

**One real piece of work per day, one commit.** The examiner may look at `git log`;
a steady history of real steps is strong evidence of your own work. Empty or trivial
commits made only for green squares are easy to spot, so don't make them.

### Day-by-day plan (26 Sep → 15 Oct)

| Date | Work | Commit message idea |
|---|---|---|
| Sat 26 Sep | Steps 1–2: tests + BM25 baseline; first EXPERIMENT_LOG entry | `Run BM25 baseline on SciFact` |
| Sun 27 Sep | `--variants dense`; chunking study (`chunking.strategy: none` vs `sentence_window`) | `Chunking study: none vs sentence window` |
| Mon 28 Sep | Step 3: full ablation + sweeps; commit `results/` | `Full four-way ablation with significance tests` |
| Tue 29 Sep | Step 4: figures; check they read in greyscale | `Generate paper figures` |
| Wed 30 Sep | Fusion method: `fusion.method: nsf` vs `rrf`, one run | `Compare rank fusion vs score fusion` |
| Thu 1 Oct | Paper §3 Method | `Paper: method section` |
| Fri 2 Oct | Paper §4 Setup + §5 results table | `Paper: setup and results table` |
| Sat 3 Oct | Step 5: failure CSV; hand-label first 20 | `Failure analysis: first 20 labelled` |
| Sun 4 Oct | Hand-label remaining 20 | `Failure analysis: all 40 labelled` |
| Mon 5 Oct | Query-subset analysis (BM25 vs dense on numeral/rare-term queries) | `Query-subset analysis` |
| Tue 6 Oct | Step 6: generate 50 answers; write the scoring rubric first | `Generate answers; scoring rubric` |
| Wed 7 Oct | Hand-score 25 answers | `Answer scoring: 25/50` |
| Thu 8 Oct | Hand-score remaining 25 | `Answer scoring complete` |
| Fri 9 Oct | Paper §6 Failure analysis + answer-quality subsection | `Paper: failure analysis` |
| Sat 10 Oct | Paper §1 Introduction + §2 Related work | `Paper: intro and related work` |
| Sun 11 Oct | Paper §7 Limitations, abstract, conclusion | `Paper: abstract, limitations, conclusion` |
| Mon 12 Oct | Slides (12) | `Presentation slides` |
| Tue 13 Oct | Fresh-clone test: clone into a new folder, follow this file from step 1 | `Reproducibility pass; README results` |
| Wed 14 Oct | Viva rehearsal with `docs/VIVA_PREP.md`; fix what you stumble on | `Viva prep notes` |
| Thu 15 Oct | Buffer | |

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

Whichever you pick, change one thing, run it, and log it.
