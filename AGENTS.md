# PoLar Independent Reconstruction

## Scope

- Reconstruct the PoLar baseline for `meta-llama/Llama-3.2-3B-Instruct`.
- Keep the base language model frozen.
- Do not add transport adapters, callbacks, low-rank adaptation, dynamic interfaces,
  multimodal retrieval, or Predictor architecture changes during the baseline phase.

## Evidence and naming

- Treat the checked-out PoLar commit as the authoritative code reference.
- Label `n_simulations=200`, `length_penalty_lambda=5.0`, and
  `random_unexplored_probability=0.1` as 2025 preliminary defaults, not confirmed 2026
  settings.
- Call the result an independent reconstruction, never an exact reproduction.
- Preserve dataset revisions, query identifiers, hashes, model revisions, prompt hashes,
  and generation configuration hashes.

## Data safety

- Deduplicate DART-Math response records by `query_id` before splitting.
- Never place one query in more than one of train, validation, and test.
- Do not oversample a human difficulty level to reach 2,000 unique questions.
- Use test labels only for final scoring or explicitly named oracle diagnostics.

## Execution gates

- The full layer path must match the original greedy forward pass before MCTS work.
- Write and run tests before implementing bug fixes or new behavior.
- Run the 20-question smoke test and report throughput, full-run time estimate, and cache
  size estimate before any full search.
- Use `gyy1` only. Check hostname, both NVIDIA A800 GPUs, memory, utilization, and active
  processes before every launch.
- Run long remote jobs in unique detached tmux sessions with unique logs and outputs.

## Git

- Work on `reconstruction/mcts-llama`, not `main`.
- Preserve upstream files and unrelated user changes.
- Commit verified milestones with conventional commit messages.
- Do not push to `tianyi-lab/PoLar`; a writable user remote is required first.
