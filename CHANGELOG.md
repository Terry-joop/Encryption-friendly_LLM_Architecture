# Changelog

All notable changes to the Encryption-friendly LLM Architecture project will be documented in this file.

## [Unreleased] - 2025-06-29

### Added
- `.gitignore` file for proper git tracking (excludes __pycache__, models, data, etc.)
- CHANGELOG.md for tracking development progress and decisions
- **ciphertext/convert_rte_eval.py**: New script for converting RTE evaluation data
  - Handles 277 RTE evaluation samples
  - Converts weights to HE-compatible format with SUBMATRIX_DIM=128
  - Provides tensor transformation for homomorphic encryption
- **ciphertext/convert_rte_train.py**: New script for converting RTE training data
  - Converts RTE training samples for HE computation
  - Mirrors eval script structure for consistency
- **ciphertext/patch_convert.py**: Utility script for weight patching (15 lines)
  - Supports model weight conversion patching
- **ciphertext/result_summary.txt**: Performance evaluation results summary
  - Documents evaluation metrics and performance metrics

### Changed
- **plaintext/requirements.txt**: Simplified dependency paths
  - Removed hardcoded conda absolute paths (e.g., `/home/conda/feedstock_root/...`)
  - Now uses simple package names for portability across different systems
  - Impact: Team members can now install directly without path conflicts
  
- **ciphertext/CMakeLists.txt**: Updated build configuration
  - Added `find_package(CUDAToolkit REQUIRED)` for explicit CUDA discovery
  - Changed CUDA_ARCHITECTURES from "all" to specific list: "80 86 89 90"
  - Reason: Support modern GPUs (Ampere, Ada architectures) more explicitly
  - Removed legacy architecture specifications (60, 61, 70, 72, 75)
  
- **ciphertext/conda/hellm-bert-env.yml**: Updated conda environment
  - Added explicit Python 3.10 specification
  - Changed CUDA channel from `nvidia/label/cuda-12.1.0` to generic `nvidia`
  - Reason: Improve compatibility across different system configurations
  
- **ciphertext/convert.py**: Improved model conversion script
  - Updated file paths to absolute paths (`/home/jovyan/Encryption-friendly_LLM_Architecture/...`)
  - Changed QKV weight handling:
    * From: Single combined query_key_value weight
    * To: Separate query, key, value weights loaded and concatenated
    * Code: `qkv_w = torch.cat([q_w, k_w, v_w], dim=0)`
  - Updated data loading paths for MRPC dataset (training and evaluation)
  - Added support for comment-out evaluation data loading (RTE format)
  
- **ciphertext/convert_rte_eval.py** (NEW FILE): RTE evaluation data conversion
  - 276-line script for converting RTE (Recognizing Textual Entailment) eval data
  - Handles weight transformation using SUBMATRIX_DIM = 128
  - Supports HE-compatible tensor format conversion
  
- **ciphertext/convert_rte_train.py** (NEW FILE): RTE training data conversion
  - Similar to eval script but for training data
  - Supports multi-layer model weight conversion
  
- **ciphertext/examples/backward-bert-multi.cpp**: Enhanced backward pass implementation
  - **GPU Configuration**: Reduced from 8-GPU to 1-GPU setup
    * Changed `const int num_gpu = 8;` to `const int num_gpu = 1;`
    * Adjusted batch_size calculation (now 16 instead of 2 per GPU)
    * Added conditional logic for single GPU configuration
    * Purpose: Enable evaluation on single GPU machines (MRPC HE eval, 1GPU modified run)
  - Added support for gradient accumulation
  - Fixed memory leak in attention computation
  - Improved numerical stability for multi-GPU training
  
- **ciphertext/examples/bert-test.cpp**: Updated test suite and dataset
  - Changed test dataset from MRPC to RTE (Recognizing Textual Entailment)
  - Updated data path: `./data_2ly_mrpc/` → `./data_2ly_rte/`
  - Updated model weights: `converted_weights_mrpc_eval.pth` → `converted_weights_rte_eval.pth`
  - Increased evaluation samples from 51 to 277 (RTE dataset has more samples)
  - Reason: RTE dataset better represents challenge in textual inference tasks
  
- **ciphertext/examples/convert2.cpp**: Updated data conversion for RTE
  - Changed dataset path: `./data_2ly_mrpc/` → `./data_2ly_rte/`
  - Updated model weights file: `converted_weights_mrpc.pth` → `converted_weights_rte.pth`
  - Now converts RTE format encrypted weights

### Technical Details

#### Problem Identified
- Original repository had machine-specific paths in requirements.txt
- Only worked on the original developer's setup
- Team members had difficulty reproducing the environment

#### Solution Applied
1. Created comprehensive .gitignore to exclude:
   - Python cache files (__pycache__, *.pyc)
   - Virtual environments (b200env/, hellm-bert-env/)
   - Large model files (pre-trained_weights/, converted_weights/)
   - Generated logs and outputs

2. Removed git tracking of __pycache__ files
   - These are generated locally and shouldn't be in version control

3. Simplified dependencies
   - Removed conda-specific file:// paths
   - Now works with standard pip install

#### Why These Changes?
- **Portability**: Code should work across different systems (Linux, Windows, Mac)
- **Collaboration**: Team members can clone and setup without manual fixes
- **Clean History**: .gitignore prevents bloat from generated files
- **Maintainability**: Future developers won't be confused by hardcoded paths
- **Dataset Diversity**: RTE provides better evaluation of inference capabilities vs MRPC
- **GPU Efficiency**: 1-GPU configuration enables testing on personal/smaller machines
- **Architecture Targeting**: Modern GPU architectures (80, 86, 89, 90) get proper support

### Tests
- [x] Requirements can be installed on clean environment
- [x] GPU configuration reduced to 1-GPU for single-machine evaluation
- [x] RTE dataset integration tested
- [x] Model weight conversion with QKV separation verified
- [x] Path conversions to absolute paths for reproducibility
- [ ] C++ compilation tested on multiple systems (TODO: Windows testing)
- [ ] Full RTE evaluation pipeline tested end-to-end (TODO: automated CI)
- [ ] Model conversion tested with different quantization schemes

### Known Issues
- GPU support may need additional setup depending on CUDA version
- Some C++ examples may need compilation adjustments for non-Linux systems
- Absolute paths in convert.py tied to specific machine setup (should use env variables)
- RTE evaluation pipeline not fully documented

---

## Previous Work Summary (Before 2025-06-29)

### Repository History (Last 20 commits)
- **2025-02-21**: Multiple updates to core components
  - Updated backward-bert-multi.cpp for multi-GPU support
  - Multiple convert.py improvements for model compatibility
  - Requirements.txt uploaded and refined
  
- **2025-02-17**: Initial repository setup and documentation

### Key Features Developed (from git history)
1. **Plaintext Architecture**: Standard LLM implementation with BERT-based models
2. **Ciphertext Architecture**: Homomorphic encryption-compatible implementations
3. **Multi-GPU Training**: Gradient accumulation and distributed computing support
4. **Model Conversion**: Tools to convert between plaintext and encrypted representations
5. **GLUE Benchmark Support**: Standard NLP evaluation capabilities

### Current Architecture
```
├── plaintext/          # Standard LLM training and evaluation
│   ├── cramming/       # Core framework
│   ├── data/           # Dataset handling
│   └── outputs/        # Training results
│
└── ciphertext/         # HE-compatible implementations
    ├── src/            # Core C++ implementations
    ├── examples/       # Usage examples
    └── test/           # Unit tests
```

---

## Development Workflow Notes

### Git Best Practices Applied
1. **Atomic Commits**: Each commit represents a logical unit of work
2. **Clear Messages**: Commit messages explain "why" not just "what"
3. **Gitignore**: Prevents accidentally tracking generated files
4. **Changelog**: Documents major changes for future reference

### Recommended Workflow for Contributors
```bash
# Before starting new work
git pull origin master
git checkout -b feature/your-feature-name

# During development - commit frequently with clear messages
git commit -m "feat: your change description

- What changed
- Why it changed
- Any known limitations or TODOs"

# When ready
git push origin feature/your-feature-name
# Create Pull Request on GitHub
```

### For Future Maintenance
- Always update this CHANGELOG when making significant changes
- Prefix commits with: feat, fix, docs, chore, refactor, test, etc.
- Keep commit messages under 50 characters for title, with detailed body below
- Link to issues/PRs when relevant: `Fixes #123`

---

**Last Updated**: 2025-06-29
**Status**: Repository re-organized and ready for collaborative development
