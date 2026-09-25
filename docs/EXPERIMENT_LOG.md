# Experiment log

Append an entry the moment a run finishes — including runs that did not work. Failed
experiments are paper content ("we tried X, it did not help, here is the number"), and
they are also the fastest way to stop yourself repeating a run you already did.

Copy the template. Do not edit old entries; add a new one instead.

---

## Template

```
### YYYY-MM-DD — <one-line description>
Commit:      <git rev-parse --short HEAD>
Config:      configs/scifact.yaml  (changed: <key>=<value>, else "unchanged")
Command:     <exact command line>
Hardware:    <GPU>, batch <n>

Result:      nDCG@10 = ...   Recall@10 = ...   Recall@100 = ...   MRR@10 = ...
Latency:     ... ms/query    Index build: ... s    Index size: ... MB

Observation: <what you actually noticed — one or two sentences>
Next:        <what this makes you want to run next>
```

---

## Entries

### 2026-__-__ — Baseline BM25, default k1/b
Commit:
Config:      unchanged
Command:     `python -m hybridqa.cli ablation --variants bm25`
Hardware:

Result:
Latency:

Observation:
Next:

---

<!-- add new entries above this line, newest last -->
