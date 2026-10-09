# Git and GitHub, from zero

Written for someone who has never used git. The setup is done; day to day you only need
the "Daily routine" section. Since 3 Oct every change goes through a pull request: the
routine for that is in [RUN_AND_PRESENT.md](RUN_AND_PRESENT.md), Section 4.

**The mental model in three sentences.** Git keeps a history of your folder on your own
computer. GitHub is a website that stores a copy of that history online. You take a
snapshot locally (*commit*), then upload the new snapshots (*push*).

---

## Part 1 — Setup (already done)

The repository exists: <https://github.com/Vivansh-07/hybrid-retrieval-qa>. Git is installed,
your name and email are set (use the email of your GitHub account, so the commits count on
your profile), and the GitHub CLI `gh` is logged in, so `git push` needs no password.

To get a working copy on another computer, follow [DEPLOY.md](DEPLOY.md), step 1.

---

## Part 2 — The daily routine

This is the whole loop. Three commands, at the end of every working session.

```cmd
cd "<your project folder>\hybrid-retrieval-qa"

git add .
git commit -m "Add BM25 baseline results for SciFact"
git push
```

**What each one does:**

| Command | What it means | Analogy |
|---|---|---|
| `git add .` | Select every changed file to go in the next snapshot | Putting documents into an envelope |
| `git commit -m "..."` | Take the snapshot, with a note about what changed | Sealing the envelope and labelling it |
| `git push` | Upload new snapshots to GitHub | Posting the envelope |

`git add` and `git commit` work offline and only touch your computer. Only `push` needs
the internet.

### Useful checks

```cmd
git status          REM what has changed since the last commit
git log --oneline   REM list your commits, newest first
```

`git status` is safe to run at any time and is the single most useful command in git.
When you're confused, run it.

### Writing commit messages

Say what changed and, when it isn't obvious, why. Present tense, no full stop.

Good:

```
Add RRF fusion and unit tests
Fix chunk-to-doc mapping double-counting overlapping chunks
Week 2 results: fusion+rerank reaches nDCG@10 of 0.71
Drop stopword removal from BM25 tokenizer; recall dropped 2 points
```

Not good: `update`, `changes`, `asdf`, `final`, `final2`, `FINAL_actually_final`.

### How often should you commit?

Once per working session, minimum. Ideally once per finished thing — a working function,
a fixed bug, a completed experiment. Your commit history is visible evidence that you did
this work yourself over four weeks, which matters for the assignment and for anyone
looking at the repo later. One giant commit dated the night before submission tells the
opposite story.

---

## Part 3 — When something goes wrong

Nothing in this section can lose your work. Read the error, find it below.

### "'git' is not recognized as an internal or external command"

Git isn't installed, or you're in a Command Prompt window that was open before you
installed it. Close it, open a new one.

### "fatal: not a git repository"

You're in the wrong folder. Run `cd "<your project folder>\hybrid-retrieval-qa"` first.
Note the quotes — the path has spaces in it, and without quotes Windows reads it as three
separate arguments.

### "Authentication failed" or "Support for password authentication was removed"

You typed your GitHub password instead of the token. Run the push again and paste the
token as the password. If Windows saved the wrong one: open **Credential Manager** (press
Win, type it) → **Windows Credentials** → find `git:https://github.com` → **Remove**. The
next push will ask again.

### "Updates were rejected because the remote contains work that you do not have locally"

The GitHub repo has something your computer doesn't — almost always because you ticked
"Add a README" when creating it. Fix:

```cmd
git pull --rebase origin main
git push
```

### "nothing to commit, working tree clean"

Not an error. It means you haven't changed anything since your last commit.

### "error: failed to push some refs" / "src refspec main does not match any"

You haven't made a commit yet. Run `git add .` then `git commit -m "Initial commit"`,
then push.

### You committed something you shouldn't have (a big data file, a token)

Undo the last commit but keep the files on disk:

```cmd
git reset --soft HEAD~1
```

Then fix the problem (add the file to `.gitignore`, delete the token) and commit again.
If you already pushed a token, go to GitHub settings and delete that token immediately —
regenerating is free, and a leaked token is not.

### "warning: LF will be replaced by CRLF"

Harmless Windows line-ending noise. Ignore it.

---

## Part 4 — Things worth doing once the repo is up

- **Add topics.** On your repo page, click the gear next to *About* and add:
  `information-retrieval`, `rag`, `bm25`, `dense-retrieval`, `cross-encoder`,
  `reranking`, `beir`, `question-answering`. This is how people find the repo.
- **Keep the README current.** Paste the ablation table in as soon as you have numbers.
  Most people who open your repo read only the README.
- **Tag the version your paper reports**, so the numbers stay findable:

  ```cmd
  git tag -a v1.0 -m "Version submitted for assessment"
  git push origin v1.0
  ```

- **Put the repo link in the paper** (Reproducibility section) and in your resume.

---

## The five commands you actually need

```cmd
git status                    REM what changed
git add .                     REM stage everything
git commit -m "message"       REM snapshot it
git push                      REM upload
git log --oneline             REM see history
```

Everything else you can look up when you need it.
