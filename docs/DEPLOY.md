# Run it on your own computer (Windows), step by step

This is a complete local deployment: a fresh copy of the code, its own Python environment,
the tests, and the web demo running on `http://127.0.0.1:7860`. Every command was run on
9–10 Oct 2026 on the project laptop (RTX 3050 Laptop GPU, Windows 11, Python 3.13) in a
new folder, `D:\Projects\hybrid-retrieval-qa`, and the outputs below are what it printed.

**Keep everything on D:.** The steps put the Python packages, pip's download cache and all
temporary files inside the project folder, so nothing is written to C:.

**You need:** Git, Python 3.12 or 3.13 (from python.org, "Add to PATH" ticked), internet for
the first run, and about 12 GB free on D:. An NVIDIA GPU makes it faster; without one,
everything runs on the CPU.

Use **Command Prompt** (or a VS Code terminal set to Command Prompt).

---

## 1. Get the code

```
D:
mkdir D:\Projects
cd D:\Projects
git clone https://github.com/Vivansh-07/hybrid-retrieval-qa.git
cd hybrid-retrieval-qa
```
While the repository is private, Git asks you to log in to GitHub the first time.

## 2. Create the project's own Python environment

```
python -m venv .venv
.venv\Scripts\activate
mkdir data\tmp data\pip-cache
set TMP=%CD%\data\tmp
set TEMP=%CD%\data\tmp
pip config --site set global.cache-dir "%CD%\data\pip-cache"
```
The prompt now starts with `(.venv)`. The two `set` lines last only for this window: run
them again if you open a new window before step 3 is finished.

## 3. Install the packages (minutes on a fast connection; about 55 minutes here)

```
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
pip install -e .
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```
Expect `GPU: True` with an NVIDIA GPU (`False` means it will run on the CPU, more slowly).
The first line downloads about 2.5 GB (32 minutes on the test run), the second about 1 GB
(22 minutes). The environment takes 5.0 GB; pip's cache (2.7 GB, `data\pip-cache`) can be
deleted once everything is installed.

## 4. Check the code

```
python -m pytest tests/ -q
```
Expect `169 passed` (about 30 s). The tests check the metrics against values worked out by hand, the
fusion and the statistics code, and that the paper's numbers match the files in `results/`.

## 5. Get the models: choose A or B

The demo needs SciFact, four models and the fine-tuned encoder. They are not on GitHub
(several GB); they live in `data\`.

**A. Copy them from a copy of the project that already has them** (a few minutes, no
internet): copy three folders from that copy's `data\` into this one's `data\`:
```
robocopy "<other copy of the project>\data\hf_cache" data\hf_cache /E
robocopy "<other copy of the project>\data\models" data\models /E
robocopy "<other copy of the project>\data\indexes" data\indexes /E
```

**B. Build them from scratch** (about 17 minutes, needs internet). This downloads SciFact and
the encoder, fine-tunes the encoder on SciFact's training claims, and chooses the fusion
weight on held-out claims, exactly as in the paper:
```
python scripts/finetune_dense.py --epochs 12
python scripts/tune_fusion.py --set dense.model=data/models/all-MiniLM-L6-v2-scifact
```
Fine-tuning prints one line per epoch, from `epoch 0 (before training): dev nDCG@10 = 0.6912`
to `epoch 12: ... dev nDCG@10 = 0.7704`, exactly the numbers in `RESULTS.md` (14.5 minutes on
the test run). Choosing the weight ends with `chosen on dev: BM25 weight 0.3 (nDCG@10 0.7879)`
(2 minutes). Both are the committed results, reproduced.

Afterwards `git status` shows `results/finetune_all-MiniLM-L6-v2-scifact/history.json` as
changed: only the training time differs (for example 655 s instead of 584 s). Keep the
committed file with
```
git checkout -- results/finetune_all-MiniLM-L6-v2-scifact/history.json
```

## 6. Start the demo

```
python scripts/demo.py --best
```
It prints `[demo] models ready in ...` and then `Running on local URL: http://127.0.0.1:7860`,
and opens that page in the browser. The very first start also downloads the reranker and the
answer model (about 1 GB; 11 minutes on the test run); after that it takes about 2 minutes.
Leave the window open; `Ctrl+C` stops it. Two warnings are harmless: "unauthenticated
requests to the HF Hub" and, on Windows, one about symlinks in the cache.

- **Live demo tab:** click one of the four claim buttons, or type a claim. Four columns show
  what BM25, dense retrieval, fusion and fusion + reranker find (✅ marks the correct paper),
  and below them the generated answer with its citation.
- **Results tab:** the ablation table, the scores of every configuration, and the figures.

**No internet in the room?** Once the models are in `data\`, add `--offline`:
`python scripts/demo.py --best --offline`.

## 7. Re-run an experiment live (optional, about 5 minutes)

To show that the numbers come from the code:
```
python scripts/run_ablation.py --variants bm25 dense fusion --out results/_live
```
It takes about 4 minutes and ends with the significance tests, identical to the committed
ones (`results/significance.json`), because retrieval is deterministic:
```
dense vs bm25     delta=+0.0227  p=0.2473    better/worse/same = 73/60/167  (not significant)
fusion vs bm25    delta=+0.0534  p=< 0.0001  better/worse/same = 76/22/202  (significant)
fusion vs dense   delta=+0.0307  p=0.0191    better/worse/same = 70/34/196  (significant)
```
Then check every number against the committed results:
```
python scripts/check_rerun.py results/_live
```
It ends with `Every metric and every significance test is identical to the committed results.`
(21 of 21 metrics per variant, 3 of 3 tests; only the time per claim differs; checked on 11 Oct
in the test folder). `results/_live` is never committed.

## 8. Update a copy you set up earlier

When the repository has changed since you set up the folder (for example, the morning of
the presentation), open Command Prompt in the folder and run:
```
.venv\Scripts\activate
set TMP=%CD%\data\tmp
set TEMP=%CD%\data\tmp
git pull
pip install -r requirements.txt
python -m pytest tests/ -q
```
`pip install` only adds packages that were added to `requirements.txt` since; for a copy
set up on 9 Oct that is `pypdfium2`, a small PDF reader the tests use (checked on 10 Oct
in the test folder). The models in `data\` are kept. The tests should pass with the count
given in step 4.

---

## What a deployment writes, and where

Everything stays inside the project folder:

| Folder | What | Size |
|---|---|---|
| `.venv\` | the Python packages | 5.0 GB |
| `data\hf_cache\` | SciFact and the downloaded models | 1.2 GB |
| `data\models\`, `data\indexes\` | the fine-tuned encoder and the passage embeddings | 150 MB |
| `data\pip-cache\`, `data\tmp\` | pip's download cache and temporary files | 2.7 GB (can be deleted) |
| `results\_live\` | output of step 7 | a few MB |

The program writes no other files, and nothing to C: (checked on the test run).

## If something goes wrong

| Problem | Fix |
|---|---|
| `python` is not recognised | Install Python from python.org with "Add to PATH" ticked, then open a new window |
| `GPU: False` | Everything still works on the CPU, more slowly; check the NVIDIA driver if you expected a GPU |
| Port 7860 is in use | `python scripts/demo.py --best --port 7861` |
| `--best needs the fine-tuned encoder` | Do step 5 (A or B) first |
| A new window has no `(.venv)` | `cd D:\Projects\hybrid-retrieval-qa` and `.venv\Scripts\activate` |
