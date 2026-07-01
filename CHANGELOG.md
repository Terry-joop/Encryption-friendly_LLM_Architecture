# Changelog

This file summarizes local changes and experiment history after cloning the repository. It is organized by date and focuses on what changed, what was tried, what went wrong, and what the current state is.

## Baseline at Clone

### 2026-05-29

- Cloned `Encryption-friendly_LLM_Architecture` locally.
- Local clone started from upstream history that already included:
  - plaintext and ciphertext implementations,
  - HEaaN external headers/libs,
  - C++ train/eval/conversion examples,
  - plaintext cramming fine-tuning/eval code,
  - MRPC-oriented HE example configuration.
- The original HE training examples were largely hardcoded by task/path and were written around the original multi-GPU workflow.

## Local Reproduction and Experiment History

### 2026-06-01 to 2026-06-04 - MRPC 1-GPU Attempt

- Ran a local MRPC HE experiment in a modified 1-GPU setup.
- Recorded MRPC eval summary in `ciphertext/result_summary.txt`:
  - `num_eval_outputs = 408`
  - `accuracy = 0.5147058824`
  - `f1 = 0.5909090909`
- Later review found this result is not a clean 1-epoch MRPC comparison:
  - `ciphertext/train_mrpc_np1.log` stops around `step: 344`.
  - MRPC 1-GPU full epoch should be about `3664` steps.
  - No complete `0epo` MRPC snapshot was found under `ciphertext/data_2ly_mrpc/0epo/`.
- Conclusion:
  - The MRPC score was likely measured from a partially trained top-level LoRA state, not from a completed epoch snapshot.
  - The score should not be compared directly to the paper's 5-epoch MRPC result.

### 2026-06-08 to 2026-06-13 - RTE Conversion and 1-Epoch Attempt

- Added/generated RTE train/eval converted data and related conversion scripts:
  - `ciphertext/convert_rte_train.py`
  - `ciphertext/convert_rte_eval.py`
  - RTE converted containers under `ciphertext/data_2ly_rte/` via symlinked converted-weight files.
- Switched local HE train/eval examples from MRPC toward RTE.
- Local RTE 1-GPU training settings used during this period:
  - task: RTE
  - data path: `./data_2ly_rte/`
  - `num_gpu = 1`
  - `batch_size = 16`
  - `one_epo_step = 2480`
  - `num_data = 2490`
  - initially tested around 1 epoch.
- Previous RTE eval log later recomputed from `ciphertext/eval_rte_np1_277.log`:
  - normal accuracy: `0.4368231047`
  - inverted binary-label accuracy: `0.5631768953`
- Findings from later review:
  - RTE 1 epoch did complete in one earlier log, but the result may have been affected by path/config issues.
  - The inverted-label result being much higher suggests a possible label/logit-order issue that must be checked when evaluating snapshots.

### 2026-06-29 - Repository Cleanup and Documentation

- Added repository-level `.gitignore` for generated/local artifacts:
  - Python caches,
  - local environments,
  - build directories,
  - converted weights,
  - generated logs/output files.
- Added and expanded `CHANGELOG.md` to document local reproduction work.
- Updated environment/build configuration for the local setup:
  - adjusted CUDA/CMake configuration,
  - updated conda environment metadata,
  - cleaned local dependency references.
- Updated conversion code to support checkpoints with separate Q/K/V projection tensors instead of a single combined `query_key_value` tensor.
- Known remaining cleanup item:
  - `ciphertext/convert.py` still contains local absolute paths and should eventually be made argument/env driven.

### 2026-06-29 - RTE 5-Epoch Retry

- Motivation:
  - MRPC and RTE earlier scores were lower than expected.
  - One hypothesis was that 1 epoch was not enough because the paper results are based on longer fine-tuning.
- Changed RTE training from 1 epoch to 5 epochs:
  - total planned steps: `2480 * 5 = 12400`
  - optimizer updates: `155` per epoch, `775` total.
- Started RTE 5-epoch run in tmux.
- Runtime observation:
  - Per-step log time is around `43-46s`, but real wall-clock progress is much slower overall.
  - One RTE epoch appears to take several days in practice on the current setup.

### 2026-06-29 - LoRA Path Bug Found and Fixed

- Found a serious path-mixing issue:
  - training/eval examples passed RTE base paths to `TransformerBlock`,
  - but LoRA encrypted weights were loaded/saved through `HEMMer::getWeightPath()`,
  - `HEMMer` defaulted that path to `./data_2ly_mrpc/`.
- Impact:
  - RTE runs could write/read LoRA weights in the MRPC directory.
  - This could explain low or inconsistent RTE/MRPC results.
- Fixes:
  - Added configurable LoRA path setters in `ciphertext/include/HELLM/HEMMer.hpp`:
    - `setWeightPath(...)`
    - `setWeightTestPath(...)`
  - Updated training to explicitly set LoRA paths to the active task weight path.
  - Updated eval so a snapshot path can be passed directly, e.g.:
    - `./build/bin/eval ./data_2ly_rte/0epo/`
- Stopped the first bad `rte_epoch5` run around `step 95 / 2480`.
- Restarted fixed RTE run in tmux as:
  - session: `rte_epoch5_fixed`
  - log: `ciphertext/train_rte_epoch5_fixed_tmux.log`
- Verified the fixed run writes current LoRA files under `ciphertext/data_2ly_rte/`.

### 2026-07-01 - Accuracy Review and Metric Sanity Checks

- Reviewed why MRPC and RTE scores were lower than paper reference values.
- MRPC review:
  - Existing MRPC log was incomplete: stopped around `step 344 / 3664`.
  - No complete MRPC `0epo` snapshot was available.
  - Existing MRPC top-level LoRA files were later overwritten by the earlier RTE/MRPC path bug, so they are no longer reliable as MRPC results.
- RTE review:
  - Previous RTE 1-epoch log appears to have reached `step 2480`.
  - Existing RTE eval result was low in normal label interpretation but improved when predictions were inverted:
    - normal accuracy: `0.4368231047`
    - inverted accuracy: `0.5631768953`
  - This suggests label/logit ordering must be checked for future evals.
- Metric note:
  - MRPC is usually reported with F1 because GLUE uses task-specific metrics.
  - RTE is reported with accuracy.

### 2026-07-01 - Hardcoding Removed from Train/Eval/Metric

- Made HE training task-selectable through environment variables in `ciphertext/examples/backward-bert-multi.cpp`.
- Supported task values:
  - `RTE` / `R`
  - `MRPC` / `M`
  - `COLA` / `C`
  - `SST2` / `T`
  - `QNLI` / `Q`
- Added runtime overrides:
  - `HELLM_TASK`
  - `HELLM_WEIGHT_PATH`
  - `HELLM_NUM_GPU`
  - `HELLM_BATCH_SIZE`
  - `HELLM_EPOCHS`
  - `HELLM_STEPS_PER_EPOCH`
  - `HELLM_TRAIN_SUBSET`
  - `HELLM_SEED`
- Made HE eval task-selectable in `ciphertext/examples/bert-test.cpp`.
- Added eval overrides:
  - `HELLM_TASK`
  - `HELLM_WEIGHT_PATH`
  - `HELLM_LORA_PATH`
  - `HELLM_EVAL_STEPS`
- Rewrote `ciphertext/metric.py` so it no longer hardcodes MRPC paths.
- Metric script now supports:
  - `--task`
  - `--pred-file`
  - `--label-file`
  - `--block-size`
  - both normal and inverted binary-label metric output.
- Verified build:
  - `cmake --build build -j --target eval train`
- Commit pushed:
  - `93e6dfd feat: make HE task selection configurable`

## Current Experiment State

### RTE Fixed 5-Epoch Run

- Active tmux session:
  - `rte_epoch5_fixed`
- Active log:
  - `ciphertext/train_rte_epoch5_fixed_tmux.log`
- Current run uses the fixed LoRA path logic that writes to `./data_2ly_rte/`.
- Important evaluation plan:
  - Wait until an epoch snapshot exists, e.g. `./data_2ly_rte/0epo/`.
  - Evaluate that snapshot directly.
  - Run metric script and compare both normal and inverted label interpretations.

Example snapshot eval after epoch 0:

```bash
cd /home/jovyan/Encryption-friendly_LLM_Architecture/ciphertext
export HELLM_KEY_PATH=./key
HELLM_TASK=RTE ./build/bin/eval ./data_2ly_rte/0epo/
```

Example metric command:

```bash
cd /home/jovyan/Encryption-friendly_LLM_Architecture
python ciphertext/metric.py --task rte --pred-file ciphertext/eval_rte_np1_277.log
```

## Current Usage Notes

### Full RTE Train

```bash
cd /home/jovyan/Encryption-friendly_LLM_Architecture/ciphertext
export HELLM_KEY_PATH=./key
HELLM_TASK=RTE ./build/bin/train
```

### MRPC Train Without Source Edits

```bash
cd /home/jovyan/Encryption-friendly_LLM_Architecture/ciphertext
export HELLM_KEY_PATH=./key
HELLM_TASK=MRPC ./build/bin/train
```

### Small Deterministic Pilot

```bash
cd /home/jovyan/Encryption-friendly_LLM_Architecture/ciphertext
export HELLM_KEY_PATH=./key
HELLM_TASK=RTE HELLM_EPOCHS=5 HELLM_TRAIN_SUBSET=256 HELLM_STEPS_PER_EPOCH=256 HELLM_SEED=42 ./build/bin/train
```

## Known Issues / Follow-Ups

- The currently running RTE full experiment is very slow; full 5 epochs may take much longer than initially expected.
- The earlier MRPC result should not be used as a paper-comparable result because the run did not complete one full epoch.
- The earlier RTE result should be rechecked using the fixed snapshot eval path and both normal/inverted metric outputs.
- `ciphertext/convert.py` still contains local absolute paths.
- `ciphertext/examples/convert2.cpp` is still task/path-specific and should eventually be made task-selectable like train/eval.
- STS-B is not yet included in the generic task selection path because its labels are floating-point while the shared train path currently assumes classification labels.
