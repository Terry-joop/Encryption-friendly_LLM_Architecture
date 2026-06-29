# Changelog

All notable changes to the Encryption-friendly LLM Architecture project will be documented in this file.

## [Unreleased] - 2025-06-29

### Added
- `.gitignore` file for proper git tracking (excludes __pycache__, models, data, etc.)
- CHANGELOG.md for tracking development progress and decisions

### Changed
- **plaintext/requirements.txt**: Simplified dependency paths
  - Removed hardcoded conda absolute paths (e.g., `/home/conda/feedstock_root/...`)
  - Now uses simple package names for portability across different systems
  - Impact: Team members can now install directly without path conflicts
  
- **ciphertext/CMakeLists.txt**: Updated build configuration
  - Modified GPU/CUDA settings for better compatibility
  - Fixed compilation flags for different C++ standards
  
- **ciphertext/conda/hellm-bert-env.yml**: Updated conda environment
  - Adjusted Python version and dependency specifications
  
- **ciphertext/convert.py**: Improved model conversion script
  - Better error handling for weight conversion
  - Support for different quantization formats
  - Code refactoring for maintainability
  
- **ciphertext/examples/backward-bert-multi.cpp**: Enhanced backward pass implementation
  - Added support for gradient accumulation
  - Fixed memory leak in attention computation
  - Improved numerical stability for multi-GPU training
  
- **ciphertext/examples/bert-test.cpp**: Updated test suite
  - Fixed type conversions (int → size_t)
  
- **ciphertext/examples/convert2.cpp**: Refactored conversion logic
  - Improved code clarity and efficiency

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

### Tests
- [x] Requirements can be installed on clean environment
- [ ] C++ compilation tested on multiple systems (TODO: Windows testing)
- [ ] Gradient computation verified numerically
- [ ] Model conversion tested with different quantization schemes

### Known Issues
- GPU support may need additional setup depending on CUDA version
- Some C++ examples may need compilation adjustments for non-Linux systems

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
