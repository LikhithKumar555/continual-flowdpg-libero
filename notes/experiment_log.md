# Experiment log

One entry per run. Write the config and seed BEFORE launching.

## Week 1 questions for supervisor (fill in answers)
1. Reward used by the provided critic (sparse success or shaped)? ->
2. What is the "initial checkpoint" (pretrained / BC on task 1 / random)? ->
3. Does the dataset contain only successful demos, or also failures / rollouts? ->
4. Is the codebase built on a Hugging Face library (LeRobot?) ->
5. Is the first suite LIBERO-Object, or LIBERO-Spatial (my notes are unclear)? ->
6. Primary per-stage budget: fixed gradient steps or fixed interaction? ->
7. Compute: GPU type/count, parallel simulators, time limits ->
8. Policy conditioning: language, task ID, or both? ->
9. Offline-only continual study, or also online interaction per stage? ->

## Run template

### YYYY-MM-DD run_id
- Goal:
- Config file / git commit:
- Method, seed, task order:
- Budget (gradient steps, rollouts):
- Result (link to cells.jsonl and heatmap):
- What I learned / next step:

## Reading notes to revisit
- IQL paper: Sections 4.1-4.3, Algorithm 1, Fig 1-2, Sec 5.3; look up tau and beta values in the appendix.
- FlowDPG: Section 3.2 (projection, shift, distillation), ablation table.
- CLARE: evaluation section (metrics definitions) - confirm exact FWT/NBT formulas.
