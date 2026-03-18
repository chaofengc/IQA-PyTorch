# Release Notes: v0.1.15

## 📢 Major Changes

### 🚩 New Metrics
- Add `afine` metric proposed in [AFINE](https://github.com/ChrisDud0257/AFINE) for No-Reference image quality assessment. Thanks to [Du CHEN](https://github.com/ChrisDud0257) for their great work 🤗, refer to their official paper for more details.
- Add `sfid`, a commonly used spatial FID metric for evaluating generative models.
- Add `maclip` metric proposed in [MACLIP](https://github.com/zhix000/MA-CLIP), introducing magnitude in IQA. Thanks to [zhicheng](https://github.com/zhix000) for their contribution 🤗.
- Add `dmm` ([paper](https://ieeexplore.ieee.org/abstract/document/10886996)), a Full-Reference IQA method using SVD-based debiased mapping to mitigate perception bias. Thanks to [baoliang](https://github.com/Baoliang93) for their great work 🤗.

### 🛠️ Enhancements
- Bundled the full Q-Align (qalign) architecture (Llama2 + mPLUG-Owl2 model implementations) directly into the package, reducing external dependencies and improving reliability. 9b94a86
- Fixed `compare2score` inference results with a dedicated mPLUG-Owl2 model variant. 79b3434
- Enhanced QAlign class with improved image processing and more robust error handling. 317888e
- Add score direction information (`lower_better` / `higher_better`) to all model cards. 4fd66f3

### 🐛 Bug Fixes
- Fix `brisque` metric when loading from local model weights. 6edfb07
- Fix GFIQA dataset path (`GFIQA` → `GFIQA-20k`). 39c89a6
- Fix spatial features used in `sfid`. 9b040e4
- Remove random permutation in `inception_score` to keep consistent and reproducible results. 812fc93
- Fix `accelerate` version constraint in requirements for compatibility. fe508fe
- Pin `transformers < 5.0` to ensure stability with Q-Align models. 83e54b7

### 📦 Build System
- Migrate from `setup.py` + `setup.cfg` + `MANIFEST.in` to modern Python packaging with `pyproject.toml` and `setuptools-scm` for dynamic versioning. 2f246e5
- Add `uv` installation option for significantly faster dependency resolution:
  ```bash
  pip install uv
  uv pip install pyiqa
  ```

### 🔧 Code Quality
- Remove unused `icecream` debug imports from multiple modules. b8e336c
- Remove `class_mapping.json` and simplify architecture registration in `__init__.py`. 317888e
- Clean up unused code and comments in mPLUG-Owl2 model classes. 317888e
- Improve `pyiqa` command line argument descriptions for better usability. 317888e

## 👥 Contributors
Thank you to the contributors who made this release possible:
- 🌟 Add `afine` metric by [@ChrisDud0257](https://github.com/ChrisDud0257)
- 🌟 Fix tensor return in `afine` by [@rockerBOO](https://github.com/rockerBOO)
- 🌟 Add `maclip` metric and `dmm` ([#287](https://github.com/chaofengc/IQA-PyTorch/pull/287)) by [@zhix000](https://github.com/zhix000)
- 🌟 Add score direction info to model cards by [@0Ky](https://github.com/0Ky)

📝 For complete details, see the [Full Changelog](https://github.com/chaofengc/IQA-PyTorch/compare/v0.1.14.1...v0.1.15)
