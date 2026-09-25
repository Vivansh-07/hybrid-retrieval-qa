# Getting this on GitHub

Everything is ready to commit — `.gitignore` already excludes the datasets, embedding
caches, LaTeX build junk and bulky run files, so nothing large or regenerable ends up
in the history.

## Fastest path (Windows)

1. Install [Git for Windows](https://git-scm.com/download/win) if `git --version` fails
   in a terminal.
2. Create an **empty** repo on GitHub named `hybrid-retrieval-qa` — no README, no
   .gitignore, no license (you already have all three).
3. Double-click `init_and_push.bat` in this folder, or from a terminal:

   ```cmd
   cd "D:\Deep Learning\Claude\hybrid-retrieval-qa"
   init_and_push.bat <your-github-username>
   ```

The script initialises the repo, makes the first commit, sets `main` as the branch, adds
the remote and pushes. Git will prompt for credentials — use a Personal Access Token as
the password, not your account password (GitHub stopped accepting passwords in 2021).

**Making a token:** GitHub → Settings → Developer settings → Personal access tokens →
Fine-grained tokens → Generate new token → give it *Contents: Read and write* on this one
repository. Copy it immediately; it is shown once.

## Manual path

```cmd
cd "D:\Deep Learning\Claude\hybrid-retrieval-qa"
git init
git add .
git commit -m "Initial commit: hybrid retrieval pipeline, evaluation harness, paper skeleton"
git branch -M main
git remote add origin https://github.com/<username>/hybrid-retrieval-qa.git
git push -u origin main
```

## Commit habits that matter for this project

Your supervisor can see the commit history, and a repo with one giant commit dated the
night before submission reads badly regardless of the code quality. Commit on every day
you work, with messages that say what changed and why:

```
git commit -m "Add RRF k sweep; k=60 is within noise of 30 and 100"
git commit -m "Fix chunk->doc collapse double-counting overlapping chunks"
git commit -m "Week 2 results: fusion+rerank at nDCG@10 0.71"
```

Tag the version the paper reports so the numbers stay findable:

```cmd
git tag -a v1.0-paper -m "Results reported in the submitted paper"
git push origin v1.0-paper
```

## Make the repo presentable

Once it is up, on the GitHub page:

- **About** → add the description and topics: `information-retrieval`, `rag`, `bm25`,
  `dense-retrieval`, `cross-encoder`, `reranking`, `beir`, `question-answering`.
- Paste the ablation table into the README's Results section as soon as you have numbers.
  A recruiter scanning the repo reads the README and nothing else.
- Add the repo URL to the paper's Reproducibility section, and the paper PDF to the repo.

## What *not* to commit

`.gitignore` handles it, but be aware: the SciFact download, the `.npy` embedding cache,
`__pycache__`, `.venv`, and LaTeX `.aux`/`.log` files all stay out. If `git status` ever
shows a file over a megabyte, stop and check why.
