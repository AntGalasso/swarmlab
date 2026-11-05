# llm_data: contract

Task items for the LLM side (`EXPERIMENT_PLAN.md` §3, §4.1): questions for MMLU, CommonsenseQA and GSM8K from the
Hugging Face Hub at pinned revisions, and the InjecAgent and PoisonedRAG records vendored in the G-Safeguard
submodule. Seeded sampling, a typed item, an answer checker per task, and the provenance of every item.
Only `adapters/hf.py` imports `datasets`.

## public.py

```python
TASKS: tuple[str, ...]          # ("mmlu", "csqa", "gsm8k", "injecagent", "poisonedrag_msmarco")

sample(task, seed, k, settings, *, part=None, root=None) -> list[TaskItem]
sources(settings, root=None) -> dict[str, dict]    # per task: dataset, config, split, revision, sha256 (manifests)
check_answer(item, answer) -> bool
render_question(item) -> str

TaskItem(task, id, question, choices, answer, source, meta)
Source(dataset, config, split, revision, sha256)
NoAnswerKey      # check_answer on a task without a deterministic checker
DatasetMismatch  # a vendored file whose sha256 differs from its pin
```

`part` ("train" or "test") restricts a vendored task to G-Safeguard's positional 80/20 split (`int(n * 0.8)` records
first); Hub tasks refuse it. `root` is the repository root that holds `vendor/` and is for tests only.

## Tasks

| task | source | pool | `question` | `answer` | `choices` |
|---|---|---|---|---|---|
| `mmlu` | `cais/mmlu`, config `all`, split `test` | 14042 | the question | letter `A`-`D` | 4 texts |
| `csqa` | `tau/commonsense_qa`, config `default`, split `validation` | 1221 | the question | letter `A`-`E` | 5 texts |
| `gsm8k` | `openai/gsm8k`, config `main`, split `test` | 1319 | the question | the number after `####`, commas removed | `()` |
| `injecagent` | `vendor/G-safeguard/TA/datasets/attack_unsucc_data.json` | 771 | the record's `User Instruction` | `""` | `()` |
| `poisonedrag_msmarco` | `vendor/G-safeguard/MA/datasets/msmarco.json` | 100 | the record's `question` | the record's `correct answer` | `()` |

`TaskItem.id` is `<task>:<position>`: the position of the row in the pinned split before shuffling (Hub), the index
in the file (InjecAgent), or the record's native key (PoisonedRAG). It is stable under the pins.

`meta` by task:

| task | keys |
|---|---|
| `mmlu` | `subject`, `answer_index` |
| `csqa` | `native_id`, `question_concept` |
| `gsm8k` | `solution` (the full worked answer) |
| `injecagent` | `record`: the whole record (`User Tool`, `Attacker Tools`, `Attacker Instruction`, `Thought`, `Tool Parameters`, `Tool Response`, ...) |
| `poisonedrag_msmarco` | `incorrect_answer`, `adv_texts` (5 poisoned passages), `native_id` |

The 771 InjecAgent records carry 17 distinct user instructions; records differ in attacker tools and instruction.
The test split of CommonsenseQA has no labels, so `validation` is used. MS MARCO passages are research use only; the
PoisonedRAG file carries its own passages and MS MARCO itself is never loaded.

## Source records

`Source.dataset`, `config`, `split` and `revision` are set on every Hub item. A vendored item has `dataset` = the
file path relative to the repository root, `config` and `revision` None, `split` = `all`, `train` or `test`, and
`sha256` = the digest of the file read with CRLF as LF (git on Windows rewrites line endings; this digest equals the
digest of the committed blob). `sources(settings)` returns the same record per task for a block's manifest.

## Draws

- Hub tasks: `datasets` runs `load_dataset(dataset, config, split=split, revision=revision)`, then
  `shuffle(seed=seed).select(range(k))`, as written in the spec. This is `datasets`' own numpy generator, pinned by
  a golden test (datasets 5.1.0, numpy 2.4.6).
- Vendored tasks: the positions of the pool ordered by one float64 uniform each, drawn in one call from
  `core.stream("llm_data", "sample", task, part or "all", seed)`; the first `k` are the draw. Golden test: torch 2.14.0.
- A draw depends on (task, seed, pinned data) only, never on the arm, the batch or `k`: a larger `k` extends the
  draw (prefix property, tested). Every arm of a seed therefore sees the same questions.
- No draw touches the torch, numpy or `random` global generators (tested for the vendored tasks).

## Answer checkers

`check_answer(item, answer)` takes the answer already extracted from the model's text.

| task | accepts |
|---|---|
| `mmlu` | a letter `A`-`D` (case-insensitive, optional parentheses or trailing `.` / `:`) or the index 0-3 as int or digit string |
| `csqa` | a letter `A`-`E`, same normalisation; integers are refused |
| `gsm8k` | a number equal to the gold number: commas, a leading `$` and a trailing `.` are ignored, `72` equals `"72.0"`; NaN, infinity and text are wrong |
| `injecagent`, `poisonedrag_msmarco` | raises `NoAnswerKey` |

A malformed answer (wrong type, empty, out of range) is wrong, not an error. `render_question(item)` is the question
followed by `A. text` lines, the layout of G-Safeguard's `format_example`. For a choice task the wrong choices are the
letters other than `answer`; the seeded wrong-answer draw belongs to the attack side.

## Invariants

- Every experiment value comes from settings; a missing key raises `core.Unset` naming it. A revision that is not a
  full lowercase 40-character commit sha (a branch name, a short sha) or a digest that is not a lowercase 64-character
  hex is refused with the key name.
- `seed` is an int >= 0 and `k` an int >= 1 (bool and float are refused); `k` larger than the pool is an error. The
  task, `seed`, `k`, `part` and the settings keys are checked before any file or Hub access.
- A vendored file whose digest differs from its pin raises `DatasetMismatch` naming the key and both digests. A
  missing file says `git submodule update --init vendor/G-safeguard`. A file of unexpected shape raises `ValueError`.
- A Hub row of unexpected shape (not 4 MMLU choices and an index 0-3, CSQA labels not `A`..`E` in order or no answer
  key, no number after `####`) raises `ValueError` naming the task and row.
- Items are frozen dataclasses of plain values; `dataclasses.asdict(item)` is JSON-able. `settings` is never mutated.
- The shuffle runs in memory (no `datasets` indices cache file); the only files written are those `load_dataset`
  caches under `HF_DATASETS_CACHE`.
- No secret is read except through the environment (below); none is printed or logged.

## Environment

| variable | use |
|---|---|
| `HF_TOKEN` | read by `datasets` / `huggingface_hub` from the environment; not needed for these public datasets |
| `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1` | rerun from the local cache; the cache is keyed by `revision`, so an offline rerun needs the same pins as the run that filled it |
| `BAZT_NETWORK_TESTS=1` | enables `tests/test_network_smoke.py` |

`adapters/hf.py` loads the repository's `.env` with python-dotenv before the first `import datasets`; variables the
process already has win. `datasets` and `huggingface_hub` read the two offline switches once at import, so set them in
the process environment when another module has imported either library earlier.

Download size of the first online load, from the dataset cards at the pinned revisions: MMLU config `all` 51.5 MB
(the 47.5 MB `auxiliary_train` split included, 3.5 MB for `test`), CommonsenseQA 1.6 MB, GSM8K 2.7 MB.

## Settings keys read

| key | value |
|---|---|
| `llm_data.revisions.mmlu`, `.csqa`, `.gsm8k` | Hub commit sha of each dataset |
| `llm_data.vendored_sha256.injecagent`, `.poisonedrag_msmarco` | sha256 of each vendored file |

`sample` reads only the keys of its task; `sources` reads all five.

## Tests

```
python -m pytest bazt_lab/features/llm_data -q
BAZT_NETWORK_TESTS=1 python -m pytest bazt_lab/features/llm_data/tests/test_network_smoke.py -q
```

Offline tests use tiny fixtures in the shapes of the pinned datasets, a fake for the Hub adapter, the real
`datasets` shuffle on an in-memory split, and the real vendored files when the submodule is checked out. The
network smoke test loads the three pinned revisions and checks item shapes (the row builders refuse unexpected
field types) and split sizes, then repeats the draw in a fresh process with both offline switches set.

## Requests

1. `experiments/profiles.toml` (runner), two tables outside any profile:

   ```toml
   [llm_data.revisions]
   mmlu = "c30699e8356da336a370243923dbaf21066bb9fe"
   csqa = "94630fe30dad47192a8546eb75f094926d47e155"
   gsm8k = "740312add88f781978c0658806c59bc2815b9866"

   [llm_data.vendored_sha256]
   injecagent = "3465082b1d5bd32f60ba5e319a305f4b05bdb75dc00189e16ed6dc2937056998"
   poisonedrag_msmarco = "d6bf508ebb0e31e09095995061bd483678e2ae8634f6ba5de1611e4adf9b1987"
   ```

   The three revisions are the Hub `main` commits of 2026-11-05 (read from the Hub API) and the values of the library
   brief. The two digests are those of the files at submodule commit `890c99f`. `tests/test_sampling.py::
   test_profiles_toml_carries_the_requested_pins` skips until the table exists and then checks these exact values.
2. `.env.example` (quality): optional `HF_TOKEN`, `HF_HUB_OFFLINE`, `HF_DATASETS_OFFLINE`.
3. `docs/EXPERIMENTS_HOWTO.md` (quality): the two commands under Tests, and the download sizes above.
4. The `llm_harness` lane's submodule must be checked out at `890c99f` for the vendored tasks; the real-file tests
   and the digests depend on it.

## Open questions

1. Calibration and evaluation questions. Seeds are disjoint (`runtime.llm_calibration_seeds` against
   `runtime.llm_seeds`), questions are not: two independent draws of a and b items from a pool of n share a*b/n
   items on average. With 60 calibration and 60 evaluation conversations: about 2.9 items for CommonsenseQA, 2.7 for
   GSM8K, 0.3 for MMLU, 4.7 for InjecAgent, 36 for PoisonedRAG (pool 100). Whether calibration questions must be
   disjoint from evaluation questions is undecided. Options: keep seed disjointness only (current); use
   `part="train"` for calibration and `part="test"` for evaluation on the vendored tasks; add an `exclude` argument
   to `sample`.
2. PoisonedRAG correctness. Upstream judges a free-text answer with an LLM (`MA/evaluate_output.py`, gpt-4o-mini).
   This feature gives the correct and the incorrect answer text and no deterministic checker; the checker, a
   normalised-containment rule or a judge call, is a harness decision.
3. MMLU download. `load_dataset("cais/mmlu", "all", ...)` fetches every split of the config (51.5 MB). Passing
   `data_files={"test": "all/test-*"}` would fetch 3.5 MB; it departs from the call written in the spec and is not
   implemented.

## Known gaps

- `tests/test_network_smoke.py` has not been run: loading the datasets downloads about 56 MB, which the build rules
  do not allow. Dataset shapes were checked against the cards at the pinned revisions and against `datasets` 5.1.0
  with those features (CommonsenseQA `choices` is `{"label": [...], "text": [...]}`; MMLU `answer` is an int), not
  against downloaded rows.
