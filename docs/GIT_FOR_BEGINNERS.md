# Git and GitHub, from zero

Written for someone who has never used git. Follow it top to bottom once; after that
you only need the "Daily routine" section.

**The mental model in three sentences.** Git keeps a history of your folder on your own
computer. GitHub is a website that stores a copy of that history online. You take a
snapshot locally (*commit*), then upload the new snapshots (*push*).

---

## Part 1 — One-time setup (about 10 minutes)

### 1.1 Install Git

Download **Git for Windows** from <https://git-scm.com/download/win> and run the
installer. Click Next on every screen — the defaults are correct.

To check it worked, open **Command Prompt** (press `Win`, type `cmd`, press Enter) and run:

```cmd
git --version
```

You should see something like `git version 2.47.0`. If you instead see
*"'git' is not recognized"*, close Command Prompt, open a new one, and try again — the
installer only adds git to new windows.

### 1.2 Tell git who you are

Git stamps your name and email on every commit. Run these two commands once, ever:

```cmd
git config --global user.name "Vivansh Garg"
git config --global user.email "vivanshgarg05@gmail.com"
```

Use the same email as your GitHub account, otherwise your commits won't be linked to
your profile — and the green contribution squares are part of why you're doing this.

### 1.3 Create the GitHub repository

1. Sign in at <https://github.com> (make an account if you don't have one).
2. Click the **+** in the top-right corner → **New repository**.
3. Repository name: `hybrid-retrieval-qa`
4. Description: *Hybrid sparse–dense retrieval with cross-encoder reranking for domain-specific QA*
5. Choose **Public** (recruiters can't see private repos, and this one is for your resume).
6. **Do not tick** "Add a README", "Add .gitignore", or "Choose a license" — your folder
   already has all three, and ticking them causes a merge conflict on your first push.
7. Click **Create repository**.

You'll land on a page with setup instructions. Ignore it; the next section is easier.

### 1.4 Make a Personal Access Token

GitHub stopped accepting your account password from the command line in 2021. You need a
token instead — think of it as a password that only works for git.

1. Go to <https://github.com/settings/tokens?type=beta> → **Generate new token**.
2. Token name: `laptop-git`
3. Expiration: 90 days (long enough for this project).
4. Repository access: **Only select repositories** → pick `hybrid-retrieval-qa`.
5. Permissions → Repository permissions → **Contents** → change to **Read and write**.
6. Click **Generate token**.
7. **Copy the token now** and paste it somewhere safe. GitHub shows it exactly once. It
   looks like `github_pat_11ABC...`.

> Treat the token like a password. Never paste it into a file inside the repo — if you
> commit a token, GitHub detects it and revokes it automatically.

### 1.5 First push

Open Command Prompt and run:

```cmd
cd "D:\Deep Learning\Claude\hybrid-retrieval-qa"
init_and_push.bat vivanshgarg05
```

The script does everything: initialises the repo, makes the first commit, and pushes.
When it asks for a **username**, type `vivanshgarg05`. When it asks for a **password**,
paste the token (right-click pastes in Command Prompt; nothing appears on screen as you
paste — that's normal). Press Enter.

Refresh your GitHub page. Your code is there.

> Windows will remember the token in Credential Manager, so you only type it once.

---

## Part 2 — The daily routine

This is the whole loop. Three commands, at the end of every working session.

```cmd
cd "D:\Deep Learning\Claude\hybrid-retrieval-qa"

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

You're in the wrong folder. Run `cd "D:\Deep Learning\Claude\hybrid-retrieval-qa"` first.
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
  git tag -a v1.0-paper -m "Results reported in the submitted paper"
  git push origin v1.0-paper
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
