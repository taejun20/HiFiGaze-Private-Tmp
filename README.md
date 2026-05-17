<div align="center">

<h1>HiFiGaze</h1>

<img src="https://img.shields.io/badge/python-3.10%2B-blue">
<img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white">
<img src="https://img.shields.io/badge/Swift-iOS-F05138?logo=swift&logoColor=white">
<img src="https://img.shields.io/badge/MediaPipe-FaceLandmarker-00A98F">
<img src="https://img.shields.io/badge/license-CC%20BY--NC%204.0-lightgrey">
<img src="https://img.shields.io/badge/paper-coming%20soon-orange">

<p><strong><em>High-fidelity appearance-based gaze estimation on smartphones, end-to-end from data collection to on-device inference.</em></strong></p>

</div>

______________________________________________________________________

## :rocket: Overview

HiFiGaze is a research codebase for **appearance-based gaze estimation on commodity smartphone cameras**. It contains the full pipeline behind three user studies, the iOS data-collection and live-inference apps, and a PC-side capture rig used for the high-resolution study.

The codebase ships:

- **Three reproducible user studies** with leave-one-subject-out training, evaluation, and result aggregation.
- **A modality zoo** of dual-eye MobileNetV4 regressors that combine RGB eye patches with optional auxiliary signals: thermal-like template (T), sclera-occluding eyelid mask (SOE), head pose (H), face-feature head-angle (FFHA), and 2D eye-corner landmarks.
- **A resolution sweep** (108 MP → QQVGA) for studying how camera resolution affects gaze accuracy.
- **iOS apps** for data collection, on-device Core ML inference with One-Euro filtering, and live demos.
- **A PC capture rig** built around an Arducam 108 MP UVC camera, with chessboard-based camera-to-eye distance estimation.

> :warning: This is **research code**. Paths, subject lists, and W&B project names are hard-coded per study; treat the numbered Python files as the canonical execution order.

______________________________________________________________________

## :file_folder: Repository Structure

```
HiFiGaze/
├── study1/                                  # 22-subject standard-res study (RGB + H/SOE/T/FFHA variants)
│   ├── 1. correct_csv.py                    # raw-CSV cleanup
│   ├── 2. merge_session.py                  # merge primary/secondary sessions & renumber frames
│   ├── 3. make_eyepatch_soe_landmarks.py    # MediaPipe-based eye-patch + SOE + landmark + GT JSON
│   ├── 3-1. make_FFHA.py                    # optional face-feature head-angle labels
│   ├── 4-{1..8}. *_train_and_eval.py        # per-modality training + leave-one-out eval
│   ├── 5. save_result_csv.py                # aggregate per-fold best CSVs
│   ├── models/                              # GazeModel{RGB,RGBT,RGBH,RGBSOE,RGBHSOE,H,HSOE,SOE,...}
│   ├── ios/                                 # CoreML conversion + iOS-friendly model variants
│   ├── helpermethods.py
│   └── run_h_full_v3.sh                     # leave-one-out launcher
├── study2-highres/                          # 108 MP / 4K study (10 subjects)
│   ├── 1-{1..3}. *_pretrain.py              # backbone pretraining on a disjoint 12-subject pool
│   ├── 2. make_lower_res_from108mp.py       # resolution sweep: 108MP→QQVGA
│   ├── 2-2. make_lower_res_from4k.py
│   ├── 3. preprocess_lower_res_from108mp.py # per-resolution eye-crop + landmark + GT preprocessing
│   ├── 4-{1..3}. *_train_and_eval_*.py      # fine-tune across all resolutions
│   ├── 5-* / 6-*                            # per-fold best-epoch selection, 4K-source experiments
│   ├── models/                              # GazeModel_{RGB,RGBT,LMK}
│   └── experiment program/
│       ├── python-pc/                       # Arducam 108 MP capture + distance estimation
│       └── study2_iOSApp/                   # phone-side capture for study 2
├── study3-bottom/                           # 10-subject bottom-camera placement study
│   └── (same numbered pipeline as study 1, narrower modality set)
├── Xcode projects/
│   ├── HiFiGaze-Collect/                    # iOS data-collection app (calibration dots + recorder)
│   ├── HiFiGaze/                            # iOS live gaze app (MediaPipe + Core ML + One-Euro)
│   ├── HiFiGaze-Demo/                       # portrait/landscape live demo
│   ├── study3_iOSApp/                       # iOS capture for study 3
│   └── BrightnessMeausureApp/               # ambient-brightness helper used during data collection
├── python-quick-test/                       # ad-hoc inference sanity scripts
├── figures/                                 # calibration patterns (ArUco, chessboard) and helpers
├── old/                                     # archived prior pipelines (Unity, Flask, baselines)
└── tmp/
```

______________________________________________________________________

## :hammer: How to Use

### Environment

The Python pipeline is tested with PyTorch + `timm`, MediaPipe Tasks (FaceLandmarker), OpenCV, and W&B. Suggested setup:

```bash
conda create -n hifigaze python=3.10 -y
conda activate hifigaze
pip install torch torchvision timm
pip install mediapipe opencv-python pandas tqdm scipy matplotlib wandb
```

The iOS apps require Xcode 16+ and CocoaPods (`pod install` inside each workspace).

Each study expects a `*-rawdata/` directory next to the scripts (e.g. `study1-rawdata/`), containing per-subject per-session frames and CSVs in the layout produced by the iOS data-collection app. The `face_landmarker.task` model file is bundled at the root of each study folder.

### Study 1 / Study 3 pipeline

Run the numbered scripts in order from inside the study folder:

```bash
cd study1                          # or: cd study3-bottom

# 1) Clean per-trial CSV (drop the first 0.5 s of each trial)
python "1. correct_csv.py"

# 2) Merge primary/secondary sessions and renumber frames
python "2. merge_session.py"
python "2-1. merge_session_ios.py"          # optional: iOS-captured data

# 3) Extract eye patches, SOE masks, eye-corner landmarks, and pixel-normalized GT
python "3. make_eyepatch_soe_landmarks.py"
python "3-1. make_FFHA.py"                   # optional: face-feature + head-angle labels

# 4) Train + leave-one-out evaluation, per modality
python "4-1. rgb_train_and_eval.py"     -g 0 -t <train ids> -v <val ids> -e <eval id>
python "4-2. rgbh_train_and_eval.py"    ...
python "4-3. rgbhsoe_train_and_eval.py" ...
python "4-4. rgbsoe_train_and_eval.py"  ...
python "4-5. h_train_and_eval.py"       ...
python "4-6. hsoe_train_and_eval.py"    ...
python "4-7. soe_train_and_eval.py"     ...
python "4-8. rgbt_train_and_eval.py"    ...

# Or run a full leave-one-subject-out sweep:
bash run_h_full_v3.sh

# 5) Aggregate per-fold best-epoch CSVs into a single results table
python "5. save_result_csv.py"
```

CLI flags accepted by every `4-*` script:

- `-g` / `--gpu` &mdash; GPU index
- `-t` / `--train` &mdash; space-separated subject IDs for training
- `-v` / `--val` &mdash; space-separated subject IDs for validation
- `-e` / `--eval` &mdash; space-separated subject IDs for held-out evaluation

Each run logs to **W&B** (project name set by the `WandB_PROJECT` variable at the top of the script) and writes checkpoints to `model_checkpoints/<project>/`. Evaluation CSVs encode the split and `ValError` directly in the filename so that `5. save_result_csv.py` can pick the best epoch per fold automatically.

### Study 2 (high-resolution) pipeline

Study 2 follows a **pretrain → resolution sweep → fine-tune** schedule:

```bash
cd study2-highres

# 1) Pretrain backbones on a disjoint 12-subject pool
python "1-1. rgb_pretrain.py"
python "1-2. rgbt_pretrain.py"
python "1-3. lmk_pretrain.py"

# 2) Generate per-resolution copies (108mp / 54mp / 8k / 6k / 4k / 2k / fhd / hd / sd / qvga / qqvga)
python "2. make_lower_res_from108mp.py"
python "2-2. make_lower_res_from4k.py"

# 3) Per-resolution eye-crop + landmark + GT preprocessing
python "3. preprocess_lower_res_from108mp.py"
python "3-1. preprocess_lower_res_from108mp_augment.py"
python "3-2. preprocess_lower_res_from4k.py"

# 4) Fine-tune the pretrained checkpoint at every resolution
python "4-1. rgb_train_and_eval_108mp.py"
python "4-2. rgbt_train_and_eval_108mp.py"
python "4-3. lmk_train_and_eval_res.py"

# 5) Select per-fold best epochs and aggregate
python "5-1-1. rgb_find_best_epochs_108mp.py"
python "5-1. rgb_result_108mp.py"
python "5-2. rgbt_result_108mp.py"
python "5-3. lmk_result.py"

# 6) 4K-source fold experiments + RGBT 4K-template preprocessing
python "6-1. rgb_train_and_eval_4k.py"
python "6-2-1. 4k_template_preprocess.py"
python "6-2-2. rgbt_train_and_eval_4k.py"
```

Reproducibility is enforced via `torch.use_deterministic_algorithms(True)` and a fixed seed (`2026`). The `RESUME_*` constants at the top of each `4-*` script let you resume an interrupted resolution sweep without re-running earlier folds.

### iOS apps

```
Xcode projects/
├── HiFiGaze-Collect/   # data collection: calibration dots, frame + GT recorder
├── HiFiGaze/           # live inference: MediaPipe FaceLandmarker → Core ML → One-Euro
├── HiFiGaze-Demo/      # portrait/landscape live demo
└── study3_iOSApp/      # study-3 specific capture variant
```

To build the live app:

```bash
cd "Xcode projects/HiFiGaze"
pod install
open HiFiGaze.xcworkspace
```

The Core ML models consumed by the live app are produced by `study1/ios/6. convert_coreML.py` and `study1/ios/6. convert_coreML_RGBT.py`, which export the iOS-friendly variants defined in `study1/ios/GazeModelRGB_IOS.py` and `study1/ios/GazeModelRGBT_IOS.py`.

`HiFiGaze-Collect` writes a directory layout that is directly consumable by `study1/2. merge_session.py`, so the same training pipeline can be re-run on freshly collected data without manual reformatting.

### PC capture rig (Study 2)

The high-resolution rig lives in `study2-highres/experiment program/python-pc/`:

```bash
cd "study2-highres/experiment program/python-pc"

# Live preview of the Arducam 108 MP camera
python arducam_demo.py -W 3840 -H 2160 -f 20 -d 1280:720

# Run the study-2 experiment manager (calibration dot + iOS sync)
python exp_manager.py

# Estimate camera-to-eye distance from a chessboard target
python camera_to_eye_distance_estimation.py
```

Camera parameters are stored in `arducam_108mp.json`.

______________________________________________________________________

## :brain: Model Zoo

All models share a dual-eye **MobileNetV4 conv-small** backbone (`mobilenetv4_conv_small.e1200_r224_in1k` from `timm`) operating on 250&times;500 eye patches, plus a small MLP for 8-D eye-corner landmarks. Variants compose additional branches on top:

| Model                | Inputs                                                                 | Where                         |
|----------------------|------------------------------------------------------------------------|-------------------------------|
| `GazeModelRGB`       | Left/right eye RGB + eye-corner landmarks                              | `study1/models`, `study2-highres/models` |
| `GazeModelRGBT`      | + thermal-like template branch (`template_cnn`)                        | `study1/models`, `study2-highres/models` |
| `GazeModelRGBSOE`    | + sclera-occluding eyelid (SOE) mask                                   | `study1/models`               |
| `GazeModelRGBH`      | + head-pose features                                                    | `study1/models`               |
| `GazeModelRGBHSOE`   | + head pose + SOE                                                       | `study1/models`               |
| `GazeModelH/SOE/HSOE`| Auxiliary-only baselines (no RGB)                                       | `study1/models`               |
| `GazeModel_LMK`      | Landmarks only (landmark-only baseline)                                 | `study2-highres/models`       |

Each variant ships in three flavours: vanilla (`_*.py`), encoder-decoder (`_ED.py`), and calibrated (`_CAL.py`). Study 1 additionally provides `_FF`, `_HA`, and `_FFHA` heads for face-feature and head-angle fusion experiments.

The output is a 2-D screen-coordinate prediction, scaled to physical centimetres inside the evaluation loop via the per-device pixel-to-cm ratio (see `evaluate_model` in any `4-*_train_and_eval.py`).

______________________________________________________________________

## :wrench: Troubleshooting

### `pod install` fails with `unknown ISA 'PBXFileSystemSynchronizedRootGroup'`

Xcode 16 introduced synchronized folder groups, which older versions of the `xcodeproj` gem can't parse. Upgrade CocoaPods and `xcodeproj`:

```bash
gem install cocoapods
gem install xcodeproj
```

(Use `sudo` if your gem directory requires it.) You need `xcodeproj` &ge; 1.25.0 and CocoaPods &ge; 1.16. Verify with:

```bash
pod --version
gem list xcodeproj
```

Then re-run `pod install`.

Alternative: in Xcode, right-click the synchronized folder in the navigator and choose **Convert to Group** to fall back to a classic `PBXGroup`.

______________________________________________________________________

## :page_facing_up: License

This codebase is released under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International). Academic and non-commercial use is permitted with attribution. For commercial licensing, please contact the authors.

______________________________________________________________________

## :link: Citation

A paper and project page are forthcoming. Once available, please cite:

```bibtex
@misc{hifigaze,
  title  = {HiFiGaze: High-Fidelity Appearance-Based Gaze Estimation on Smartphones},
  author = {TBD},
  year   = {TBD},
  note   = {Paper and project page: TBD}
}
```

- :page_with_curl: Paper: _TBD_
- :globe_with_meridians: Project page: _TBD_
- :movie_camera: Demo video: _TBD_
