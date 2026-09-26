# Task: group the SAE features of OLMo-2-1B into concept groups

You are the **orchestrator**. You will launch **subagents** that read feature labels and group them, then merge their
groups in rounds, until there is one set of concept groups. A helper script validates every step. Read this whole
document before doing anything.

**Scope of this run: PILOT — shards 00, 01, 02, 03 only** (about 32k features). Do not touch other shards.

---

## 1. Background (why this exists)

- We trained **sparse autoencoders (SAEs)** on the residual stream of every layer (0–15) of the language model
  OLMo-2-1B. Each SAE has 32,768 **features**; each feature is a direction the model uses, and it fires on specific
  tokens in text.
- Every feature has an **auto-interpretability label**: one sentence written by an LLM after looking at the text the
  feature fires on most, e.g. `the word "restore" and its variants in contexts of bringing back or renewing something`.
- Features are named `L<layer>:<index>`, e.g. `L8:123` = feature 123 of layer 8.
- **Goal:** group features that detect **the same concept** into one group, **across all layers**. The same concept
  often appears as a separate feature at several layers (e.g. a "Texas" feature at L5, another at L8, another at
  L12). Those belong in one group. This mirrors the "supernodes" Anthropic uses in their attribution graphs (e.g. a
  "Texas" supernode made of several Texas features), but built globally instead of by hand for one prompt.
- The groups will be used in a visualization tool to show what the model computes, and to steer the model with a
  whole group at once. So groups must be **specific and meaningful**, not topic buckets.

## 2. What a good group is (the most important section)

A group = **one concept that a person would name in a few words**, e.g. "Texas", "the possessive 's", "Python import
statements", "LaTeX commands for including figures", "the word 'restore'", "flatulence and defecation".

**Put features in the same group when they detect the same thing**, even if:
- they come from different layers (this is the whole point),
- their labels are worded differently (`the word 'received' or its forms` and `forms of the verb "receive"` → same),
- one label is slightly broader or narrower than the other but clearly about the same core thing,
- the concept appears in different languages (English "not" and Russian "не" as negation → same group).

**Keep features in different groups when they only share a domain or topic:**
- `Python import statements` ≠ `Python for-loops` ≠ `Python function definitions` (all Python, different concepts)
- `hex color codes in CSS` ≠ `CSS numeric values with px units` (both CSS, different concepts)
- `baseball game situations like bases and downs` ≠ `football club abbreviations` (both sports, different concepts)
- `the word "star" referring to celestial bodies` ≠ `the word "star" meaning a celebrity` (same word, different sense)

**Surface-form features** (a specific token, substring, or punctuation) group only with features for the **same**
surface form: `the substring "Mc" at the start of names` is its own group; it does not join `surnames` or
`Scottish names`. `ellipses of three dots` is its own group, not `punctuation`.

**Never use broad buckets.** These are all wrong as groups: "code", "programming", "math", "numbers", "places",
"animals", "grammar", "science", "web development", "misc". If you are about to create a group whose name would fit
hundreds of unrelated features, split it into the actual concepts.

**Rule of thumb:** if naming the group honestly needs the word "or" between two unrelated things, it is two groups.

**Group names:** 5–15 words, as specific as a good label, saying exactly what the member features detect.
Good: `the word "Texas" and references to the state of Texas`, `LaTeX \includegraphics and figure sizing commands`,
`possessive apostrophe-s after names and nouns`. Bad: `Texas`, `LaTeX`, `possessives` (too short), `various
programming-related technical terms in many contexts` (vague).

**Size:** there is no target. Expect many small groups (2–10 members) and some singletons (a feature whose concept
appears nowhere else in your input). Large groups (50+) are fine only if every member really is the same concept
(e.g. dozens of features for "the word 'the'"). Never pad a group with loosely related features to make it bigger.

**Noisy labels:**
- Vague labels (`technical terms and concepts in programming contexts`, `words in various contexts`) that you
  cannot place in a specific concept: put them in **one** group per shard named `vague labels: <what they have in
  common>`, e.g. `vague labels: generic programming vocabulary`. Do not let them pollute specific groups.
- Garbled or empty-looking labels: one group named `unclear or garbled labels`.

## 3. Files and the helper script

Everything lives in `results/OlMo2_1b/sae_resid_topk/agent_grouping/` (below: `AG/`). **Write nothing outside
`AG/`. Never modify `AG/shards/`.** Run all commands from the repository root with the venv python:

```
venv/bin/python -m evoke.OlMo2_1b.interp.agent_grouping check <level>
venv/bin/python -m evoke.OlMo2_1b.interp.agent_grouping finalize <level>
```

| path | who writes it | contents |
|---|---|---|
| `AG/shards/shard_XX.tsv` | already there | input: one feature per line, `L8:123<TAB>label`, ~8,094 lines, a random mix of all 16 layers |
| `AG/level0/shard_XX.jsonl` (or `shard_XX.part1.jsonl`, `shard_XX.part2.jsonl`, …) | level-0 subagents | one group per line: `{"name": "...", "members": ["L8:123", "L3:9001", ...]}` |
| `AG/level<k>/job_YY.jsonl` (or `job_YY.part1.jsonl`, …), k ≥ 1 | merge subagents | one group per line: `{"name": "...", "from": ["lv0.03.17", "lv0.01.5", ...]}` |
| `AG/index/level<k>_ZZ.tsv` | `check` | one group per line: `key<TAB>size<TAB>layers<TAB>name<TAB>sample label | sample label` — what merge subagents read |
| `AG/jobs/level<k>.json` | `check` | `{"job_00": ["level0_00.tsv", "level0_01.tsv"], ...}`: which index files each merge job combines |
| `AG/members/`, `AG/groups.json` | `check` / `finalize` | resolved memberships and the final result; never edit by hand |

Group keys look like `lv0.03.17` = line 17 (0-based, counting only non-empty lines, all part files of that unit in
sorted file-name order) of unit 03's level-0 output. You never compute keys yourself: always copy them from the index
files.

**The rules `check <level>` enforces** (it prints exact errors and exits 1 if anything is wrong):
- level 0: every feature id of the shard appears in exactly one group's `members`; no ids that are not in the shard.
- level k ≥ 1: every group key of the job's input index files appears in exactly one group's `from`; no unknown keys.
- every line is valid JSON with a non-empty `name`.

## 4. Procedure

### Step 1 — level 0: group each shard (4 subagents in parallel)

Launch one subagent per shard (00, 01, 02, 03) at the same time, each with the **level-0 subagent prompt** below
(fill in `XX`). Wait for all of them.

### Step 2 — check level 0

Run `check 0`. If it fails, send the error lines to the subagent of that shard (or a new one with the same prompt plus
the errors) to fix **only** the listed ids, then run `check 0` again. Repeat until it prints `level 0 OK`.
The output also prints the merge jobs of level 1 (written to `AG/jobs/level1.json`).

### Step 3 — merge rounds

For level k = 1, 2, …: read `AG/jobs/level<k>.json`, launch one subagent per job in parallel with the **merge
subagent prompt** below (fill in `k`, `YY`, and the job's input index files), wait, run `check <k>`, fix errors the
same way. When `check <k>` reports `1 units`, go to step 4. For the pilot: level 1 has 2 jobs, level 2 has 1 job.

If a merge job's input index files together have more than ~15,000 lines, stop and report instead of launching it.

### Step 4 — finalize and report

Run `finalize <k>` with the last level. Then write `AG/REPORT.md`:
- the numbers `finalize` and the last `check` printed (groups, features, groups per feature, size distribution,
  groups spanning more than one layer),
- the 30 largest groups: name, size, layers, 5 member labels (look them up in the shards),
- 30 random groups that span 3+ layers, same format,
- your honest assessment: are the groups specific concepts or did they drift broad? examples of the best and the worst
  groups, and anything that went wrong.

## 5. Level-0 subagent prompt (copy it, fill in XX)

> You are grouping features of sparse autoencoders trained on the language model OLMo-2-1B. Read
> `GROUPING_TASK_AGENT_PROMPT.md` sections 1–3 in the repository root first: they define what a good group is, and you
> must follow them exactly.
>
> **Your input:** `results/OlMo2_1b/sae_resid_topk/agent_grouping/shards/shard_XX.tsv`, about 8,094 lines, one
> feature per line: `<feature id><TAB><label>`. Read the whole file (it is long: read it in consecutive chunks with the
> Read tool's offset/limit, e.g. 2,000 lines at a time, until you have seen every line). Do not use scripts or code to
> group: the grouping is your judgment from the labels.
>
> **Your task:** put every feature of the file into exactly one group (section 2). Work concept by concept: first skim
> everything to see which concepts recur, then build the groups, then go through the file again line by line and make
> sure every id is placed.
>
> **Your output:** `results/OlMo2_1b/sae_resid_topk/agent_grouping/level0/shard_XX.jsonl`, one group per line,
> `{"name": "<5-15 word name>", "members": ["L8:123", ...]}`. Copy ids exactly. If the output is too long for one
> write, split it over `shard_XX.part1.jsonl`, `shard_XX.part2.jsonl`, … (each group on exactly one line, in exactly
> one part). Write nothing else anywhere.
>
> **When done**, run `venv/bin/python -m evoke.OlMo2_1b.interp.agent_grouping check 0` from the repository root. If it
> lists errors for shard_XX, fix exactly those (place missing ids into the right groups, remove duplicates or ids that
> are not in your shard) and run it again until shard_XX has no errors. Errors for other shards are not yours. Reply
> with the number of groups you made and 5 example groups.

## 6. Merge subagent prompt (copy it, fill in k, YY, and the input files)

> You are merging concept groups of sparse-autoencoder features of OLMo-2-1B. Read `GROUPING_TASK_AGENT_PROMPT.md`
> sections 1–3 in the repository root first: they define what a good group is, and you must follow them exactly.
>
> **Your input:** the index files `<file A>` and `<file B>` in
> `results/OlMo2_1b/sae_resid_topk/agent_grouping/index/`. Each line is one existing group:
> `<key><TAB><size><TAB><layers><TAB><name><TAB><sample label> | <sample label>`. The groups were made independently
> from different random slices of the features, so the same concept usually exists in both files under slightly
> different names. Read both files completely (in chunks with offset/limit if long).
>
> **Your task:** combine the groups of both files into one set of groups: groups that are the same concept (section 2)
> become one group; everything else stays its own group. Merging is only for **the same concept**: never absorb a
> specific group into a broader one (`Python import statements` does not merge into `Python code`), never merge two
> groups just because they share a domain, and never create broad buckets. When you merge, write a name (5–15 words)
> that covers exactly the merged concept, no broader. Every input key must end up in exactly one output group,
> including groups that merge with nothing.
>
> **Your output:** `results/OlMo2_1b/sae_resid_topk/agent_grouping/level<k>/job_YY.jsonl`, one output group per line,
> `{"name": "<5-15 word name>", "from": ["<key>", "<key>", ...]}` (a group that merges with nothing has one key). Copy
> keys exactly from the index files. If too long for one write, split into `job_YY.part1.jsonl`, `job_YY.part2.jsonl`,
> …. Write nothing else anywhere. Do not use scripts or code to decide merges.
>
> **When done**, run `venv/bin/python -m evoke.OlMo2_1b.interp.agent_grouping check <k>` from the repository root and
> fix any errors listed for job_YY (missing keys, duplicate keys, unknown keys) until job_YY has none. Reply with the
> number of input groups, the number of output groups, and 5 example merges.

## 7. Rules for you, the orchestrator

- Only the pilot shards (00–03). Launch level-0 subagents in parallel; merge jobs of one level in parallel.
- Never edit the subagents' output yourself except to fix `check` errors when a subagent could not; never edit
  `AG/shards/`, `AG/index/`, `AG/members/`, `AG/jobs/`, or `AG/groups.json` by hand.
- Never write files outside `AG/`, never change code in the repository, never delete anything outside `AG/`.
- If something is unclear or broken (a script error that is not a `check` validation error, a subagent that keeps
  failing), stop and write what happened into `AG/REPORT.md` instead of improvising.
