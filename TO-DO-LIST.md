# PoLar Reconstruction To-Do List

- [x] Read the requested PoLar, CoLa, and DART-Math sources.
- [x] Fix the upstream PoLar commit and create `reconstruction/mcts-llama`.
- [x] Restore the upstream `search_space.png` blob interrupted during cloning.
- [x] Add a writable user Git remote for milestone pushes.
- [x] Capture source, dependency, and GPU manifests.
- [x] Build and validate the 7,500-query public model-relative difficulty split.
- [ ] Reproduce the frozen LLaMA baseline.
- [ ] Validate full, skip, repeat, segment-repeat, and joint layer paths.
- [x] Implement diagnostic and Predictor-compatible MCTS with cache replay checks.
- [x] Add batched path execution, two-GPU sharding, and append-only label outputs.
- [ ] Run the 20-question smoke test and publish time and storage estimates.
- [ ] Run the ReDM-1 pilot and validation-only parameter calibration.
- [ ] Generate supervision, train five Predictors, and evaluate online pass at 1 through 5.
