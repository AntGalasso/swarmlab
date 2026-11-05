# gnn: G-Safeguard GNN reference, LLM side (spec 2.7)

The published GNN of G-Safeguard (Wang et al., ACL 2025), ported to torch_geometric 2.8.1, trained on our own LLM
conversation logs with the upstream setup, and run as the `GNN` arm: flag attacked agents, cut all their outgoing
edges. Upstream code is read from `vendor/G-safeguard` only by parity tests (`adapters/upstream.py`); nothing is copied.

## public.py

| name | what |
|---|---|
| `ARMS` | `GNN` (calibrated=False, the published 0.5 cut) and `GNN_recalibrated` (calibrated=True, statistic `gnn_logit`, acts when logit > `ctx.thresholds['gnn_logit']`, +inf while calibrating); both `platforms=("llm",)` |
| `GNNArm` | the `core.Defender` |
| `log_conversations(settings, unit, records)` | append conversation records of a `gnn_train` unit (checked first, fsynced) |
| `read_conversations(settings, family)` | all logged records of a family (a repeated id kept once) |
| `train_unit(unit, settings, embed=None)` | train the unit's family GNN once its whole family is logged; raises `TrainingDataIncomplete(family, have, need)` before that. Returns the checkpoint metadata. Repeating a finished training returns it without redoing it |
| `train_family(settings, family, records, embed=None)` | the same on explicit records |
| `checkpoint_path(settings, family)` | `<gnn.artifacts_dir>/gnn_<family>.pt` |
| `TrainingDataIncomplete` | see `train_unit` |
| `deploy_family(attack, task, settings)` | the one rule for the GNN of a cell (D105): its own family if in `gnn_train.families`; else `gnn.fallback_family` `pi_same_task` -> `insider_pi__<task>`, `none` -> `None` (cell refused); unset -> `core.Unset`. The arm, the runner preflight and llm_harness all call it |
| `training_finished(settings, family)` | True iff the family's checkpoint is marked finished and its history has `gnn_train.epochs` entries. `GNNArm.reset` raises `RuntimeError` naming `train_unit` for any other checkpoint (a crash leaves a partial one); the harness calls `train_unit` again on relaunch, which resumes |

`train_unit(..., embed=...)` takes an injectable embedder (llm_harness passes a fake one when `llm.api == "fake"`, so no MiniLM download).

Conversation record (one JSON line): `{id, attack, task, backbone, adj [N][N] of 0/1 (adj[i][j] = 1: i sends to j),
attackers [indices], turns [rounds][N] of the text each agent wrote that round}`. Family = `<attack>__<task>`.
The llm_harness writes it with `log_conversations` while running `gnn_train` units.

## The model

`adapters/pyg_model.py`: `GATwithEdgeConv` (message = alpha * (x_j + W_e e), extra `lin_edge_attr`, returns
`(out, W_e e)`, `edge_dim` mandatory) and `MyGAT`, parameter names equal to upstream's, so state dicts load both ways.
Tested against a hand-computed graph, an independent loop oracle, and the upstream classes loaded from the submodule.
Refusals that upstream crashes on: `concat=False`, `hidden_dim % heads != 0`.

Embeddings: `all-MiniLM-L6-v2` at the pinned revision, loaded once, batched, cached by text hash. `normalize_embeddings`
is False as upstream; the model's own pipeline already ends in a Normalize module, so vectors are unit length.
Graph tensors follow upstream: edge (i, j) carries agent j's per-turn embeddings; node feature = mean of the first turn
over incoming edges (`"target"`, used by upstream validation and deployment; `gnn_train.train_node_index` picks
`"source"`, which upstream's `train()` uses).

## Training (`gnn_train`)

Adam(lr, weight_decay), CosineAnnealingLR(T_max, eta_min) per epoch, BCEWithLogitsLoss, batches of graphs, best
validated epoch saved (`torch.save` of `{state_dict, meta}`, atomic, `weights_only` loadable). After every epoch the full resumable state (model, optimizer, scheduler, epoch, history, best accuracy, RNG state) is written atomically to `<gnn.artifacts_dir>/epochs/<unit id>/epoch_NNNN.pt`; the newest `gnn_train.keep_checkpoints` files, the best epoch and the final one are kept; resume loads the newest file that loads completely. Differences from
upstream (deliberate): first epoch always kept; all randomness keyed; per-epoch resumable state; metadata saved.

## The arm

Per round: embed `obs.texts`, run the GNN on the conversation so far over `ctx.adj`, flag agents with
`sigmoid(logit) >= gnn.cut`, flags persist until the next conversation (`obs.round_in_conversation == 1`, `obs.t == 1` when it is None) when `gnn.carry_flags`, deny
every attempted delivery of a flagged sender. If the intervention budget cannot pay, the most suspicious senders are
funded first. Reports `statistics["gnn_logit"]` float64 `[B, N]`. State: `flagged`, `history` (texts).

## Settings read

`gnn.cut`, `gnn.carry_flags`, `gnn.fallback_family` (`pi_same_task` | `none`; read only for a cell outside `gnn_train.families`; owner key, unset -> `core.Unset`), `gnn.artifacts_dir`,
`gnn.embedding.{model,revision,batch_size}`; `gnn_train.{logs_dir, keep_checkpoints, families, agents, rounds, backbone,
conversations_per_setting, edge_probabilities, attackers, hidden_dim, num_heads, num_layers, dropout, aggr_type,
epochs, lr, weight_decay, batch_size, scheduler_t_max, scheduler_eta_min, train_fraction, split, train_node_index,
seed}`. Costs from `ctx.costs["deny"]`. No default in code; a missing key raises `core.Unset`.

## Invariants

No ground truth is read. Scores float64, masks bool. Results per episode do not depend on B. `reset` is once per
episode (seed run); conversations inside it are separated by `obs.round_in_conversation == 1`.

## Open questions

1. DECIDED (D106): `GNN` keeps the published 0.5; `GNN_recalibrated` is the second arm. `gnn.recalibrate` no longer exists.
2. DECIDED (D105): the arm loads the GNN of the cell's own (attack, task) family from `ctx.cell`. For infection and AiTM cells `gnn.fallback_family` decides: `pi_same_task` uses `insider_pi__<task>`, `none` refuses the cell. Unset until the owner sets it.
3. Backbone for training logs (`gnn_train.backbone`) and whether the GNN is budget-capped (spec section 11). Here it
   is uncapped in the grid; if run under a cap, denies are funded most-suspicious first.
4. DECIDED (D99): cut as `deny` per round, the flag kept until the conversation ends (not `edge_quarantine`).
5. Hyperparameter values (`gnn_train.hidden_dim` etc.): upstream defaults are 1024, 8 heads, 2 layers, dropout 0.2,
   20 epochs, lr 0.001, weight decay 0.0002, batch 32, cosine T_max 10, eta_min 1e-5, train fraction 0.8, `mean`
   aggregation as `main` script (upstream `MyGAT` default). Not set in the repository yet.

## Requests (this feature edits no other file)

0. core `EpisodeContext.cell` (now present): the llm_harness must fill `{'attack', 'task'}`. The arm refuses a context without it.
1. core `Observation`: add `texts: Optional[List[List[str]]] = None` (`[B][N]` the text each agent wrote this round,
   delivered or not). The arm reads `getattr(obs, "texts", None)` and refuses without it. The LLM platform fills it.
2. LLM platform: `obs.round_in_conversation == 1` at the first round of every conversation; `Platform` calls `reset(ctx)` once
   per seed run. The platform writes training logs with `log_conversations` for `gnn_train` units and calls
   `train_unit` (after the last unit of a family).
3. `experiments/profiles.toml`: add `[gnn]` with `cut = 0.5`, `carry_flags = true`, `fallback_family` (unset, owner key),
   `artifacts_dir`, `[gnn.embedding]` (`model = "sentence-transformers/all-MiniLM-L6-v2"`, `revision =
   "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"`, `batch_size`), and to `[gnn_train]`: `logs_dir`, `hidden_dim`,
   `num_heads`, `num_layers`, `dropout`, `aggr_type`, `epochs`, `lr`, `weight_decay`, `batch_size`, `scheduler_t_max`,
   `scheduler_eta_min`, `train_fraction`, `split`, `train_node_index`, `seed`. Values are the owner's (question 5).
4. `experiments/profiles.toml`: `gnn_train.keep_checkpoints` (integer >= 1; unset, the runner adds a value). The epoch files live under `<gnn.artifacts_dir>/epochs/` for the sync feature to upload.
5. pyproject: `python-dotenv` is imported by `adapters/sbert.py`; declare it if not a dependency already.

## Known gaps

No end-to-end run with the real MiniLM model (no network use in tests); the arm has not been exercised by the real
LLM platform (not yet built against `obs.texts`).
