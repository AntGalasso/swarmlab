# gnn: G-Safeguard GNN reference, LLM side (spec 2.7)

The published GNN of G-Safeguard (Wang et al., ACL 2025), ported to torch_geometric 2.8.1, trained on our own LLM
conversation logs with the upstream setup, and run as the `GNN` arm: flag attacked agents, cut all their outgoing
edges. Upstream code is read from `vendor/G-safeguard` only by parity tests (`adapters/upstream.py`); nothing is copied.

## public.py

| name | what |
|---|---|
| `ARMS` | `GNN` (calibrated=False, the published 0.5 cut, D106) and `GNN_recalibrated` (calibrated=True, statistic `gnn_logit`, acts when logit > `ctx.thresholds['gnn_logit']`, +inf while calibrating; calibrated at alpha on attack-free conversations, D106, D107); both `platforms=("llm",)` |
| `GNNArm` | the `core.Defender`; needs `ctx.cell` = {'attack', 'task'} and `obs.texts`, else `ValueError` |
| `GNNRecalibrated` | `GNNArm` with the logit cut (`recalibrated = True`) |
| `log_conversations(settings, unit, records)` | append conversation records of a `gnn_train` unit to `<gnn_train.logs_dir>/<family>/units/<unit id>.jsonl` (checked first, fsynced) |
| `read_conversations(settings, family)` | all logged records of a family, files in name order (a repeated id kept once, its first occurrence) |
| `train_unit(unit, settings, embed=None)` | train the unit's family GNN once its whole family is logged; raises `TrainingDataIncomplete(family, have, need)` before that. A family is whole at `gnn_train.conversations_per_setting` x `len(gnn_train.edge_probabilities)` x `len(gnn_train.attackers)` conversations (800 per family, D82); more raises `ValueError`, and so does a family outside `gnn_train.families`. Returns the checkpoint metadata. Repeating a finished training returns it without redoing it |
| `train_family(settings, family, records, embed=None)` | the same on explicit records |
| `checkpoint_path(settings, family)` | `<gnn.artifacts_dir>/gnn_<family>.pt` |
| `TrainingDataIncomplete` | see `train_unit` |
| `deploy_family(attack, task, settings)` | the one rule for the GNN of a cell (D105): attack `none` (attack-free) -> the unique `gnn_train.families` entry with that task, whatever the fallback key (zero or several: `ValueError` naming the task); otherwise its own family if in `gnn_train.families`; else `gnn.fallback_family` `pi_same_task` -> `insider_pi__<task>`, `none` -> `None` (cell refused); unset -> `core.Unset`. The arm and the runner preflight (`launch.py`) call it |
| `training_finished(settings, family)` | True iff the family's checkpoint is marked finished, its history has `gnn_train.epochs` entries, its recorded settings fingerprint (gnn_train keys and embedding) equals the current one and, when conversations are logged for the family, its recorded conversation ids equal the logged ones. `GNNArm.reset` raises `RuntimeError` naming `train_unit` otherwise |
| `train_pending(settings, embed=None)` | trains, or resumes, every family in `gnn_train.families` whose conversations are all logged and whose `training_finished` is False; returns the metadata of the trainings it ran, `[]` (a no-op) when nothing is pending. The runner calls it at the end of every `gnn_train` launch |

`train_unit(..., embed=...)` and `train_pending(..., embed=...)` take an injectable embedder. For offline runs set the test-only key `gnn.embedding.fake = true` (documented for tests and e2e only; unset means false): `shared_embedder` then returns a deterministic 384-wide encoder (`fake_encode`, vectors keyed by the text) and nothing is downloaded. llm_harness does not pass an embedder.

Conversation record (one JSON line): `{id, attack, task, backbone, adj [N][N] of 0/1 (adj[i][j] = 1: i sends to j),
attackers [indices], turns [rounds][N] of the text each agent wrote that round}`. Family = `<attack>__<task>`.
The llm_harness writes it with `log_conversations` while running `gnn_train` units. Its `id` is
`<unit id>:<conversation>`, and it also writes `question` and `distinct_questions`, which the trainer does not read.
Its questions are outside every evaluation and calibration question (D104). `gnn_train.agents`, `rounds` and `backbone`
must match every record (`check_logged`).

## The model

`adapters/pyg_model.py`: `GATwithEdgeConv` (message = alpha * (x_j + W_e e), extra `lin_edge_attr`, returns
`(out, W_e e)`, `edge_dim` mandatory) and `MyGAT`, parameter names equal to upstream's, so state dicts load both ways.
Tested against a hand-computed graph, an independent loop oracle, and the upstream classes loaded from the submodule.
Refusals that upstream crashes on: `concat=False`, `hidden_dim % heads != 0`; `num_layers < 1` and an `aggr_type` other
than `last` or `mean` are refused as well.

Embeddings: `all-MiniLM-L6-v2` at the pinned revision (`adapters/sbert.py`), loaded once per model, revision and batch
size, batched, cached by text hash. `normalize_embeddings` is False as upstream; the model's own pipeline already
ends in a Normalize module, so vectors are unit length.
Graph tensors follow upstream: edge (i, j) carries agent j's per-turn embeddings; node feature = mean of the first turn
over incoming edges (`"target"`, used by upstream validation and deployment; `gnn_train.train_node_index` picks
`"source"`, which upstream's `train()` uses).

## Training (`gnn_train`)

Adam(lr, weight_decay), CosineAnnealingLR(T_max, eta_min) per epoch, BCEWithLogitsLoss, batches of graphs, best
validated epoch saved (`torch.save` of `{state_dict, meta}`, atomic, `weights_only` loadable). After every epoch the full resumable state (model, optimizer, scheduler, epoch, history, best accuracy, RNG state) is written atomically to `<gnn.artifacts_dir>/epochs/<unit id>/epoch_NNNN.pt`; the newest `gnn_train.keep_checkpoints` files, the best epoch and the final one are kept; resume loads the newest file that loads completely. Differences from
upstream (deliberate): first epoch always kept; all randomness keyed; per-epoch resumable state; metadata saved.

Split (`gnn_train.split`): `sequential` trains on the first `round(n x train_fraction)` graphs (upstream's order),
`shuffled` on the same number of graphs from a permutation keyed by seed and family. Progress rows are appended to
`<gnn.artifacts_dir>/train_state/units/<unit id>.jsonl`, and `<unit id>.done` is written at the end. A finished training
is returned, not repeated; a finished unit whose checkpoint names another unit id raises `RuntimeError`. The unit id
(`gnn-<family>-<12 hex>`) hashes the family, the gnn_train settings, the embedding and the record ids.

## The arm

Per round: embed `obs.texts`, run the GNN on the conversation so far over `ctx.adj`, flag the active agents with
`sigmoid(logit) >= gnn.cut` (GNNRecalibrated: `logit > ctx.thresholds['gnn_logit']`). With `gnn.carry_flags` a flag
persists until the next conversation (`obs.round_in_conversation == 1`, `obs.t == 1` when it is None), D99. Every
attempted delivery of a flagged sender is denied. If the intervention budget cannot pay, the most suspicious senders are
funded first (D94), and a sender that does not fit is skipped. The arm sets no `threshold_restricted`, so every deny
counts toward alpha (D108). Reports `statistics["gnn_logit"]` float64 `[B, N]`, the logits of the conversation so far.
State: `flagged`, `history` (texts).

## Settings read

`gnn.cut`, `gnn.carry_flags`, `gnn.fallback_family` (`pi_same_task` | `none`; read only for a cell outside `gnn_train.families`; owner key, unset -> `core.Unset`), `gnn.artifacts_dir`,
`gnn.embedding.{model,revision,batch_size}`, `gnn.embedding.fake` (test and e2e only, unset means false),
`compute.device`; `gnn_train.{logs_dir, keep_checkpoints, families, agents, rounds, backbone,
conversations_per_setting, edge_probabilities, attackers, hidden_dim, num_heads, num_layers, dropout, aggr_type,
epochs, lr, weight_decay, batch_size, scheduler_t_max, scheduler_eta_min, train_fraction, split, train_node_index,
seed}`. Costs from `ctx.costs["deny"]`. No default in code; a missing key raises `core.Unset`.

## Invariants

No ground truth is read. Scores float64, masks bool. Results per episode do not depend on B. `reset` is once per
episode (seed run); conversations inside it are separated by `obs.round_in_conversation == 1`.
Float32 (D117): the model weights, the embeddings and the forward pass are float32 (torch default). The lane does not
read `compute.dtype`, so a float64 reference run still computes the GNN in float32; `gnn_logit` is returned as float64.

## Open questions

1. Backbone for training logs (`gnn_train.backbone`) and whether the GNN is budget-capped (spec section 11). Here it
   is uncapped in the grid (`GNN` is not in `llm.capped_arms`); if run under a cap, denies are funded most-suspicious
   first (D94). The value is the owner's: profiles.toml leaves `gnn_train.backbone` unset (commented out), and the
   trainer and the gnn_train unit grid read it.
2. `gnn.fallback_family` (`pi_same_task` | `none`) for infection and AiTM cells. D105 sets the rule, not the value. The
   value is the owner's: profiles.toml leaves it unset (commented out), so those cells refuse until it is set.
3. Hyperparameters (D82, upstream setup). The upstream values are set in profiles.toml `[gnn_train]`. The owner's
   `gnn_train.split` (`sequential` | `shuffled`) and `gnn_train.seed` are unset (commented out).

## Requests (this feature edits no other file)

None open. The core fields (`EpisodeContext.cell`, `Observation.texts`, `Observation.round_in_conversation`) exist and
the llm_harness fills them; the gnn settings are in profiles.toml (see Open questions for the unset keys);
`python-dotenv` is declared in pyproject.toml.

## Known gaps

No end-to-end run with the real MiniLM model (no network use in tests). The arm is tested on its own with a fake encoder
(`tests/test_arm.py`, `tests/test_device.py`); no test runs `GNNArm` inside the LLM platform, and the llm_harness tests
replace the gnn functions with stubs.

`train_pending` first checks every family's logs against `gnn_train.agents/rounds/backbone` (`logs.check_logged`) and refuses foreign records naming the file and record id before anything trains. Epoch and checkpoint files are replaced with `core.atomic_replace` (retries a transient Windows PermissionError).
