# princeton continual factoid memorization: data

pools for the benchmark in `original_research/lscl/princeton_benchmark.md` (what the datasets are, procedure, rules, our plan).
this file is only how to build the files and what is in them.

# running it

from repo root:
```
python -m datasteps.LSCL.princeton.download_all
```
or one at a time: `python -m datasteps.LSCL.princeton.popqa` etc. downloads are cached in `raw/`, reruns are offline.

output: `data/datasteps/LSCL/princeton/`
```
kvr.jsonl               4,000   synthetic, no download
popqa.jsonl            14,267   hf akariasai/PopQA test.tsv (via hf-mirror, plain http)
triviaqa.jsonl         86,481   hf mandarjoshi/trivia_qa rc.nocontext train+validation parquet (via hf-mirror), deduped per question_id, 2 conflicting ids dropped
lama.jsonl             34,039   T-REx from dl.fbaipublicfiles.com/LAMA/data.zip, 41 relations
entityquestions.jsonl  22,075   nlp.cs.princeton.edu entity-questions dataset.zip, test split, 24 relations
raw/                            the downloaded tsv / zips / parquet
```

# fact schema

one json object per line:
```
{
  "id": "popqa_4222362",            # "<source>_<source id>", unique within the file
  "question": "What is X's occupation?",
  "answer": "politician",           # canonical target string
  "aliases": ["politician", ...],   # accepted strings for exact-match eval, always contains answer
  "subject": "George Rankin",       # entity the fact is about, null when the source has none (triviaqa, entityquestions)
  "relation": "occupation",         # wikidata pid, popqa prop label, "kvr", or "trivia"
  "meta": {...}                     # source specific, see below
}
```
`meta` per source:
- kvr: `{}`
- popqa: `prop_id, s_uri, o_uri, s_pop, o_pop` (monthly wikipedia views, the long-tail signal; sort by s_pop ascending for long tail)
- triviaqa: `split, answer_type`
- lama: `relation_label, template, sub_uri, obj_uri`. question is the cloze template with `[X]` filled and `[Y]` -> `___`. 5,275 rows have the blank mid-sentence (e.g. "[X] plays [Y] music .")
- entityquestions: `split`

these are the unfiltered, model-agnostic pools. the per-model "does the base model already know it" filter and the
train-time prompt format are in the plan at the bottom, not here.
