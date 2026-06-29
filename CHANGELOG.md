# Changelog

All notable changes to the Encryption-friendly LLM Architecture project are documented here.

This file is intended to record both upstream history and the local reproduction changes made after cloning the repository.

## [Unreleased] - 2026-06-29

### Local Repository Timeline

- **2026-05-29**: Repository cloned from `https://github.com/Donghwan-Rho/Encryption-friendly_LLM_Architecture`.
  - Clone landed at commit `4a2df0f` on `master`.
  - At clone time, the upstream repository already contained plaintext and ciphertext implementations, HEaaN external headers/libs, C++ examples, conversion scripts, and plaintext cramming code.
- **2026-06-29 `564fb0c` - `chore: add .gitignore`**
  - Added top-level `.gitignore`.
  - Removed tracked generated Python cache files from the repository history at this point.
  - Exclusion scope includes Python caches, local virtual environments, generated outputs, model/checkpoint artifacts, converted HE weights, build directories, logs, and temporary files.
- **2026-06-29 `8c917aa` - `docs: add comprehensive CHANGELOG`**
  - Added this `CHANGELOG.md` file.
- **2026-06-29 `179994d` - `refactor: update dependencies and build configurations`**
  - Main functional/code update for the local reproduction.
  - Added RTE conversion scripts, updated build/environment files, changed MRPC-oriented HE examples toward RTE, and changed training from the original 8-GPU setup to a local 1-GPU setup.
- **2026-06-29 `0895609` - `docs: update CHANGELOG with GPU configuration change`**
  - Documented the 1-GPU configuration change.
- **2026-06-29 `97e3fee` - `docs: comprehensive CHANGELOG update with detailed code changes`**
  - Expanded the changelog with more detailed file-level notes.
- **2026-06-29 current update**
  - Rewrote this changelog to explicitly include step, epoch, data-count, batch-size, and clone-to-current history details.

### Added

- **`.gitignore`**
  - Added repository-level ignore rules for generated files and large local artifacts.
  - Important ignored categories:
    - `__pycache__/`, `*.pyc`, Python build metadata.
    - Local environments such as `b200env/` and `hellm-bert-env/`.
    - HE build directories such as `ciphertext/build/`.
    - Converted weights and generated data outputs.
    - Logs and temporary files.

- **`CHANGELOG.md`**
  - Added a structured record of repository changes and local reproduction decisions.
  - Later expanded to include exact step/epoch/GPU changes.

- **`ciphertext/convert_rte_eval.py`**
  - New RTE evaluation conversion script.
  - Uses `DIM = 768`, `HIDDEN_DIM = 3072`, `MAX_SEQ_LEN = 128`, `NUM_HEADS = 12`, `BATCH_SIZE = 16`.
  - Uses `SUBMATRIX_DIM = 128` with an explicit caution that this is not 64.
  - Converts model weights into HE-compatible block layouts.
  - Adds RTE eval inputs/masks to a TorchScript-style container.
  - Intended output is the RTE eval converted-weight container used by `bert-test.cpp`.

- **`ciphertext/convert_rte_train.py`**
  - New RTE training conversion script.
  - Mirrors the eval conversion structure for train data.
  - Converts RTE train inputs/masks and model weights for HE fine-tuning.

- **`ciphertext/patch_convert.py`**
  - Added a small one-off patch helper for `ciphertext/convert.py`.
  - Replaces old combined `query_key_value.weight` access with separate `query.weight`, `key.weight`, and `value.weight` loading followed by `torch.cat(...)`.

- **`ciphertext/result_summary.txt`**
  - Added local result note for MRPC HE eval on the 1-GPU modified run.
  - Recorded values:
    - `num_eval_outputs = 408`
    - `accuracy = 0.5147058823529411`
    - `f1 = 0.5909090909090907`
    - true label counts `{0: 129, 1: 279}`
    - prediction counts `{0: 203, 1: 205}`

### Changed

#### `plaintext/requirements.txt`

- Removed machine-specific conda build paths from dependency entries.
- Changed packages from `package @ file:///...` style to standard package names.
- Affected packages:
  - `antlr4-python3-runtime`
  - `hydra-core`
  - `omegaconf`
  - `packaging`
  - `PyYAML`
  - `typing_extensions`
- Purpose: make plaintext environment installation less tied to the original machine.

#### `ciphertext/CMakeLists.txt`

- Added explicit CUDA Toolkit discovery near the top:
  - `find_package(CUDAToolkit REQUIRED)`
- Changed CUDA architecture targeting.
  - Before: broad/legacy setting including `all` and older real architectures such as 60, 61, 70, 72, 75.
  - After: `80 86 89 90`.
- Purpose: target newer NVIDIA GPU architectures more directly for the local CUDA setup.
- Note: the file still contains another `find_package(CUDAToolkit REQUIRED)` later in the file, so CUDA discovery is duplicated.

#### `ciphertext/conda/hellm-bert-env.yml`

- Changed CUDA channel:
  - Before: `nvidia/label/cuda-12.1.0`
  - After: `nvidia`
- Added explicit Python version:
  - `python=3.10`
- Purpose: make the conda environment more explicit and easier to solve on the local setup.

#### `ciphertext/convert.py`

- Replaced placeholder paths with local absolute paths:
  - `weight_path = "/home/jovyan/Encryption-friendly_LLM_Architecture/plaintext/pre-trained_weights/test/model.safetensors"`
  - `save_path = "/home/jovyan/Encryption-friendly_LLM_Architecture/ciphertext/converted_weights/converted_model.pt"`
- Updated MRPC train data loading paths from relative placeholder-style paths to absolute local paths:
  - `plaintext/fine-tuning_data/mrpc_train_inputs`
  - `plaintext/fine-tuning_data/mrpc_train_masks`
- Updated commented MRPC eval path examples to absolute local paths.
- Changed attention weight loading:
  - Before: read a single combined `attn.self_attention.query_key_value.weight`.
  - After: read separate `query.weight`, `key.weight`, and `value.weight`, then concatenate:
    - `qkv_w = torch.cat([q_w, k_w, v_w], dim=0)`
- Purpose: support checkpoints whose Q/K/V projections are stored as separate tensors.
- Known limitation: the absolute paths are machine-specific and should eventually be replaced with CLI args or environment variables.

#### `ciphertext/examples/backward-bert-multi.cpp`

This file contains the most important step/epoch/runtime behavior changes.

- Changed dataset/weight directory:
  - Before: `./data_2ly_mrpc/`
  - After: `./data_2ly_rte/`

- Changed task:
  - Before: `auto task = "M";` for MRPC.
  - After: `auto task = "R";` for RTE.

- Changed GPU configuration:
  - Before: `const int num_gpu = 8;`
  - After: `const int num_gpu = 1;`

- Changed effective batch size through the existing formula:
  - Formula remains `static_cast<int>(16.0 / num_gpu)`.
  - Before with 8 GPUs: `batch_size = 2`.
  - After with 1 GPU: `batch_size = 16`.

- Changed training data count:
  - Before: `const int num_data = 3668;` for MRPC.
  - After: `const int num_data = 2490;` for RTE.

- Changed one-epoch step count:
  - Before: `const HELLM::u64 one_epo_step = 458;` for MRPC on the original 8-GPU setup.
  - After: `const HELLM::u64 one_epo_step = 2480; // temp`.
  - Important: this differs from the comment/README value `RTE: 310`.
  - Reasoning from the current code:
    - With `num_gpu = 1`, each loop step consumes one sample index via `rank + num_gpu * step`.
    - RTE train data count is `2490`.
    - `2480` is divisible by `batch_size = 16`, producing `155` optimizer updates per epoch.
    - This leaves 10 RTE train samples unused in the current loop.
  - Documentation impact: any README or run note that still says RTE uses `310` steps is outdated for this 1-GPU local training mode.

- Changed epoch count:
  - Before: `for (HEaaN::u64 epo = 0; epo < 5; epo++)`.
  - After: `for (HEaaN::u64 epo = 0; epo < 1; epo++)`.
  - Current local run is configured for 1 epoch, not 5.

- Changed converted-weight file:
  - Before: `converted_weights_mrpc.pth`.
  - After: `converted_weights_rte.pth`.

- Changed labels:
  - Before: active labels were `HELLM::ModelArgs::MRPC_LABELS`.
  - After: active labels are `HELLM::ModelArgs::RTE_LABELS`.

- Added single-GPU optimizer handling:
  - If `num_gpu == 1`, rank 0 now sequentially runs:
    - `optimizerStep_bert(task, num_step)` for layer 0.
    - `optimizerStep_bert(task, num_step)` for layer 1.
    - `optimizerStep_head_bert(task, num_step)`.
    - `optimizerStep_head2_bert(task, num_step)`.
  - If `num_gpu != 1`, the previous rank-split optimizer behavior remains:
    - ranks `< 3` handle one layer path.
    - ranks `3..5` handle the other layer path.
    - rank `6` handles `head`.
    - remaining rank handles `head2`.

- Optimizer-step numbering remains:
  - `num_step = one_epo_step * epo / batch_size + step / batch_size`.
  - With current values (`one_epo_step=2480`, `batch_size=16`, `epo=0`), this gives optimizer step numbers `0..154`.

#### `ciphertext/examples/bert-test.cpp`

- Changed eval data path:
  - Before: `./data_2ly_mrpc/`
  - After: `./data_2ly_rte/`
- Changed converted eval-weight file:
  - Before: `converted_weights_mrpc_eval.pth`.
  - After: `converted_weights_rte_eval.pth`.
- Changed evaluation loop count:
  - Before: `for (int j = 0; j < 51; ++j)`.
  - After: `for (int j = 0; j < 277; ++j)`.
- Meaning:
  - The example moved from a 51-step MRPC smoke/eval configuration to a 277-step RTE eval configuration.
  - The code still uses `rank + size * j` indexing, so total consumed records depend on MPI size/rank behavior.

#### `ciphertext/examples/convert2.cpp`

- Changed converted input container:
  - Before: `./data_2ly_mrpc/converted_weights_mrpc.pth`.
  - After: `./data_2ly_rte/converted_weights_rte.pth`.
- Changed plaintext output directory:
  - Before: `./data_2ly_mrpc/`.
  - After: `./data_2ly_rte/`.
- Purpose: generate HE plaintext/materialized weight files for the RTE path.

### Baseline Upstream History Before Local Changes

- **2025-02-10 `5c89ce2` - first commit**
  - Added the initial full repository.
  - Included root README, `plaintext/`, `ciphertext/`, HEaaN headers/libs, CMake files, C++ examples, conversion code, tests, and cramming plaintext training/eval code.
- **2025-02-17 `44dc1af`**
  - Updated plaintext architecture/configuration and fine-tuning data saving logic.
- **2025-02-18 commits**
  - `6ce43ae`: README update in `ciphertext/`.
  - `7d5975c`, `e12609b`, `d59fd12`: iterative `ciphertext/convert.py` updates.
  - `709f527`: small `backward-bert-multi.cpp` update.
- **2025-02-21 commits**
  - Several root README updates and a merge from upstream master.
- **2025-03-11 `f5cba6f`**
  - Added root `requirements.txt`.
- **2025-03-13 `73b5444`**
  - Commented out an unused part of `ciphertext/examples/convert2.cpp`.
- **2025-03-14 `506b1fa`**
  - Moved/refined requirements under `plaintext/requirements.txt`.
  - Added Python 3.11 generated cache files, later removed by local `.gitignore` cleanup.
- **2025-03-17 commits**
  - Two small `ciphertext/README.md` updates.
- **Clone baseline**
  - Local clone on 2026-05-29 started from `4a2df0f`, before the local 2026-06-29 cleanup/RTE/1GPU changes.

### Current Runtime Configuration Summary

Current `ciphertext/examples/backward-bert-multi.cpp` local training configuration:

```cpp
const std::string weight_pth = "./data_2ly_rte/";
const int num_gpu = 1;
const int batch_size = static_cast<int>(16.0 / num_gpu); // 16
auto task = "R";
const HELLM::u64 one_epo_step = 2480; // temp
const int num_data = 2490; // temp
for (HEaaN::u64 epo = 0; epo < 1; epo++)
```

Current `ciphertext/examples/bert-test.cpp` local eval configuration:

```cpp
static const std::string data_path = "./data_2ly_rte/";
torch::jit::load(data_path + std::string("converted_weights_rte_eval.pth"));
for (int j = 0; j < 277; ++j)
```

### Known Documentation Gaps

- `ciphertext/README.md` still documents the original/general 8-GPU-oriented setup:
  - `num_gpu = 8`
  - `batch_size = 2`
  - `one_epo_step = 310 # RTE`
- That README does not yet describe the current local 1-GPU RTE training mode:
  - `num_gpu = 1`
  - `batch_size = 16`
  - `one_epo_step = 2480`
  - `epo < 1`
- `convert.py` currently contains absolute `/home/jovyan/...` paths.
- `CMakeLists.txt` has duplicated `find_package(CUDAToolkit REQUIRED)`.

### Verification Notes

- Git worktree was clean before this changelog rewrite.
- Current branch: `master`, tracking `origin/master`.
- Detailed history source:
  - `git log --reverse --date=short --pretty=format:'%h %ad %an %s' --stat`
  - `git reflog --date=iso`
  - direct inspection of current `backward-bert-multi.cpp`, `bert-test.cpp`, `convert2.cpp`, `convert.py`, and conversion scripts.

---

**Last Updated**: 2026-06-29
**Status**: Local clone history and current 1-GPU RTE step/epoch configuration documented.

## Follow-up Experiment - 2026-06-29

### RTE 1-GPU Training Retry With 5 Epochs

- Motivation:
  - Previous local RTE/1-GPU experiment used `epo < 1` and produced lower-than-expected accuracy.
  - This retry tests whether the reduced epoch count contributed to the accuracy drop.

- Code change:
  - File: `ciphertext/examples/backward-bert-multi.cpp`
  - Changed training epoch loop from:
    - `for (HEaaN::u64 epo = 0; epo < 1; epo++)`
  - To:
    - `for (HEaaN::u64 epo = 0; epo < 5; epo++)`

- Configuration kept unchanged from the local RTE/1-GPU setup:
  - `weight_pth = "./data_2ly_rte/"`
  - `num_gpu = 1`
  - `batch_size = 16`
  - `task = "R"`
  - `one_epo_step = 2480`
  - `num_data = 2490`

- Expected training schedule:
  - `2480` HE training steps per epoch.
  - `5` epochs total.
  - `12400` HE training loop steps total.
  - Optimizer updates occur every `16` steps.
  - Expected optimizer update count: `155` per epoch, `775` total.

- Runtime note:
  - Previous log showed individual HE steps can take tens of seconds, so this run may take a long time.
  - Experiment result and final accuracy should be appended here after the run completes.
