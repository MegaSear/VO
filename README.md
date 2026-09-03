# Monocular Visual Odometry on Low-Parallax Aerial Imagery

Comparative evaluation of three two-frame pose estimators — **SIFT+Essential**, **SIFT+LK (optical flow)**, and **SuperPoint+LightGlue** — on the [ALTO](https://arxiv.org/abs/2207.12317) UAV dataset, with validation-set
hyperparameter tuning and an explicit failure-mode analysis.

📄 **Full write-up:** [`tex/article.pdf`](tex/article.pdf)

## TL;DR

After tuning each estimator's hyperparameters on a held-out validation split
(rather than using library defaults), both classical estimators reconstruct the
test trajectory almost exactly (0.26–0.34% relative ATE), while
SuperPoint+LightGlue — despite competitive per-pair rotation accuracy — accumulates
~6.1% relative ATE because of a single severe pose-estimation failure early in the
chained sequence. We show this failure mode is:

- **joint**, not independent: on every degenerate pair, rotation and translation
  error blow up together (never one without the other) — expected, since both
  come from one SVD decomposition of the same Essential matrix;
- associated with **low inter-frame parallax and low Essential-matrix inlier ratio**
  (Spearman ρ = −0.64 and −0.48 respectively), and not with the scene-planarity
  statistic ($R_H$) that a naive reading of the geometry would suggest — though
  with only 56 test pairs this is a correlational, diagnostic finding, not a
  controlled causal test;
- visible in trajectory-level drift (ATE) far more than in per-pair error medians
  — a *windowed* ATE (resetting the chain every few steps) shows all three
  estimators are comparable on typical pairs, and the gap is concentrated in one
  bad window.

| Estimator            | ATE (realistic, %) | $e_R$ median (°) | Latency (ms) | Peak VRAM (MB) | Median CPU RAM (MB) |
| --------------------- | ------------------- | ------------------ | ------------- | --------------- | --------------- |
| SIFT+Essential        | 0.34%                | 0.177               | 279.6         | 0                | 27.6             |
| **SIFT+LK**           | **0.26%**            | **0.139**           | **108.6**     | 0                | 22.8             |
| SuperPoint+LightGlue  | 6.08%                | 0.275               | 198.8         | 532.6            | 0                |

(gap=5, 56 test pairs, tuned hyperparameters — see the write-up for the full
protocol, grids searched, and caveats.)

## Repository contents

```
.
├── scripts/
│   ├── run_experiment.py      # CLI entry point — orchestrates the full pipeline
│   └── config.py               # ExperimentConfig, GRIDS, FALLBACK_PARAMS
├── src/
│   ├── datasets/
│   │   └── alto.py             # AltoDataset — frame loading, GT pose, splits
│   ├── estimators/
│   │   ├── base.py             # BaseEstimator — shared geometric back end
│   │   ├── sift_essential.py
│   │   ├── sift_lk.py
│   │   └── superpoint_lightglue.py
│   ├── evaluation/
│   │   ├── pose_eval.py        # PoseEvaluator — per-pair and trajectory-level runs
│   │   ├── ate.py               # ATE / windowed ATE analysis + trajectory plots
│   │   ├── statistics.py        # get_stat / get_auc / get_rpe (return data, not prints)
│   │   └── failure_analysis.py
│   ├── optimization/
│   │   └── grid_search.py       # validation-split hyperparameter search
│   ├── geometry/
│   │   ├── epipolar.py
│   │   ├── metrics.py
│   │   └── texture.py
│   └── profiling/
│       └── profiler.py          # EstimatorProfiler — latency / CPU & GPU memory
├── figures/generated/            # ATE plots, written by scripts/run_experiment.py
├── tex/article.pdf                # write-up: math, protocol, results, discussion, limitations
├── requirements.txt
└── README.md
```

## Pipeline overview (`scripts/run_experiment.py`)

1. **`AltoDataset`** (`src/datasets/alto.py`) — loads ALTO RGB frames + GPS-INS
   ground-truth pose, splits a contiguous 200-frame segment into train/val/test
   (40/30/30, split by time, not shuffled).
2. **Estimators** (`SiftEssentialEstimator`, `SiftLkEstimator`,
   `SuperPointLightGlueEstimator`) — share one geometric back end
   (`BaseEstimator._pose_recovery`): Essential-matrix RANSAC → `cv2.recoverPose`,
   plus a homography-vs-fundamental degeneracy check (an ORB-SLAM-style
   model-selection score, not a standalone planarity metric) and a parallax
   estimate, so any accuracy difference between estimators is attributable to
   the detector/matcher front end only.
3. **Hyperparameter search** (`src/optimization/grid_search.py`) — grid search
   on the validation split (RANSAC threshold, feature/keypoint budget, ratio-test
   threshold per estimator), selecting by median per-pair pose error.
4. **`PoseEvaluator`** (`src/evaluation/pose_eval.py`) — runs an estimator over
   a split either as an independent-pair sequence (`run_as_sequence`, for
   RPE/AUC@K) or a chained trajectory (`run_as_trajectory`, for ATE), including
   a *windowed* ATE variant that periodically resets the chain to isolate single
   catastrophic pairs from long-horizon drift.
5. **Failure-mode analysis** (`src/evaluation/failure_analysis.py`) — flags
   per-pair rotation/translation failures against fixed thresholds and
   correlates them with parallax, inlier ratio, and planarity.
6. **Profiling** (`src/profiling/profiler.py`) — latency and peak CPU/GPU
   memory per estimator, over a small (5-pair) sample; see caveat below.

**A note on naming:** the pipeline reports a `repeatability` field per pair, kept
under that name for continuity with earlier iterations of the code, but it is
not classical multi-view detector repeatability. It's a per-estimator
correspondence/match retention rate, and the underlying event differs by
estimator (ratio-test survival, optical-flow tracking success, LightGlue match
acceptance) — see inline comments in each estimator for the exact definition
used in each case.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` pins LightGlue to a specific commit (`lightglue @ git+https://github.com/cvg/LightGlue.git@<hash>`)
for reproducibility, plus `torch`, `torchvision`, `kornia`, `opencv-python`, `numpy`,
`pandas`, `scipy`, `matplotlib`, `tqdm`, `psutil`.

The pipeline expects the ALTO frames and `query.csv` under `<home>/frames/`; set
the `home` path in `scripts/config.py::ExperimentConfig`.

## Reproducing the results

```bash
python scripts/run_experiment.py
```

Useful flags for faster iteration:
```bash
python scripts/run_experiment.py --skip-search     # use FALLBACK_PARAMS instead of grid search
python scripts/run_experiment.py --skip-profile     # skip latency/memory profiling
python scripts/run_experiment.py -v                  # debug-level logging
```

The key parameters controlling the canonical protocol live in
`scripts/config.py::ExperimentConfig` (see `tex/article.pdf`, Sec. 2, for why
these values were chosen):
```python
split_weights = {"train": 0.4, "val": 0.3, "test": 0.3}
test_gap, test_step = 5, 1     # sequence protocol (RPE/AUC@K/failure rate)
# ATE: gap = stride = 5 (non-overlapping chained steps)
```

Latency/memory numbers above come from one fixed profiling run
(`warmup=2, runs=5` on a 5-pair sample); expect a few percent of run-to-run
jitter in the CPU timings on a re-run, with the ranking preserved.

## Known limitations (see `tex/article.pdf`, Sec. 5, for full discussion)

- Small test set (56 pairs / 12 chained ATE steps) — individual pairs carry
  disproportionate weight.
- Hyperparameters were selected via a single grid-search pass on one fixed
  validation split; they were not cross-validated across different split
  boundaries or seeds, so the tuned values may be somewhat specific to this
  particular 60-frame validation segment.
- The test split was consulted more than once during exploratory analysis before
  the protocol above was fixed as canonical; a strict confirmatory evaluation
  would need a fresh, never-seen segment.
- No fine-tuning of SuperPoint/LightGlue weights was attempted — the train split
  is currently unused.

## References

- Cisneros et al., *ALTO: A Large-Scale Dataset for UAV Visual Place Recognition
  and Localization*, arXiv:2207.12317.
- Lindenberger et al., *LightGlue: Local Feature Matching at Light Speed*, ICCV 2023.
- Umeyama, *Least-Squares Estimation of Transformation Parameters Between Two
  Point Patterns*, IEEE TPAMI 1991.
- Mur-Artal et al., *ORB-SLAM: A Versatile and Accurate Monocular SLAM System*,
  IEEE Trans. Robotics, 2015 (source of the homography-vs-fundamental
  model-selection score used for the degeneracy check).