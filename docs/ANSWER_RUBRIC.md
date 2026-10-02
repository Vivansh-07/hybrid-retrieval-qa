# Answer-scoring rubric

Written on 1 Oct 2026, **before** any of the scored answers were generated, so the
rules cannot be bent to fit them. Do not change it after you start scoring; if a case
is unclear, decide it with the closest rule and write the decision in `notes`.

## What is scored

50 test claims × 2 readers = 100 answers:

- **generative**: `flan-t5-base` writes an answer from the top 5 passages and cites them as [1]–[5]
- **extractive**: `roberta-base-squad2` copies a span from the top passage

Both readers get the same 5 passages, from the full pipeline (fusion + reranking).

**Scoring is blind as far as possible.** `results/answer_scoring.csv` lists the 100
answers in a random order and does not say which reader wrote which; the key is in a
separate file that `--summarize` uses afterwards. Don't open the key before you finish.
Caveat, added after generating (1 Oct): with the original prompt, 49 of the 50 generative
answers were only a citation such as `[3].`, which gave the reader away. On 2 Oct the
generative reader was changed to explain-then-cite (prompt chosen on training claims, see
the experiment log), so both readers now return text; the sheet also drops the trailing
`[n]` and shows the cited passage in its own column for both. The two can still sometimes
be told apart (extractive answers are short spans, generative ones full sentences), so
the blinding is partial. The rubric itself was not changed. State this in the paper.

## The four criteria (0, 1 or 2 each)

Judge against the **gold abstract** shown in the sheet (the paper the claim is about)
and the **cited passage** text, not against your own knowledge.

### 1. Correctness: does the answer get the claim right?
| Score | Meaning |
|---|---|
| 2 | States what the gold abstract says about the claim (supports it, refutes it, or the key finding) |
| 1 | Partly right: right topic and direction but vague, hedged, or missing the key point |
| 0 | Wrong, off-topic, or no content |

### 2. Completeness: is the answer useful on its own?
| Score | Meaning |
|---|---|
| 2 | A reader understands the finding without opening the paper (direction + the key qualifier, e.g. population, dose, effect size) |
| 1 | Some useful content, but a reader would need the paper to understand it |
| 0 | No useful content |

### 3. Faithfulness: is every statement supported by the passages it was given?
| Score | Meaning |
|---|---|
| 2 | Everything it says is in the cited passage(s) |
| 1 | Mostly supported, with a minor detail that is not |
| 0 | States something the passages do not say, or contradict |

### 4. Citation accuracy: does the cited passage actually support the answer?
| Score | Meaning |
|---|---|
| 2 | The cited passage contains the support for the answer |
| 1 | It is on the right topic but does not contain the specific support |
| 0 | Unrelated, or no citation |

## Rules for the tricky cases (decided in advance)

- **No content** (the answer is only a citation such as `[3]`, or empty): correctness 0,
  completeness 0, citation accuracy scored normally, and **faithfulness left blank** (it
  states nothing, so there is nothing to check). Write `no_content` in `notes`.
  The summary reports how many answers had no content and averages faithfulness over
  the others.
- **"Not enough evidence"**: if the gold abstract is **not** among the five passages
  (`gold_in_passages` = no), this is the right behaviour: correctness 2, completeness 1,
  faithfulness 2, citation accuracy 2. If the gold abstract **is** among them, it is a
  miss: correctness 0, completeness 0, faithfulness 2, citation accuracy 0.
- **Extractive answers are short spans**: score them by the same rules. A correct span
  without context usually gets completeness 1.
- **Two citations, one good and one bad**: citation accuracy 1.
- Score each answer on its own; do not compare it with other answers to the same claim.

## Procedure

1. `python -m hybridqa.cli answer --limit 50` and
   `python -m hybridqa.cli answer --limit 50 --set answer.mode=extractive` (already run for you).
2. `python scripts/answer_sheet.py` builds `results/answer_scoring.csv` (blind, shuffled).
3. Open it in Excel, fill the four score columns (0/1/2) and `notes`. Save as CSV.
   Commit after every 25 answers.
4. `python scripts/answer_sheet.py --summarize` checks every score, joins the hidden key,
   and writes `results/answer_quality.json`: the mean of each criterion per reader.
5. Report: n = 100 answers (50 per reader), one annotator (you), this rubric, 0–2 scale.
