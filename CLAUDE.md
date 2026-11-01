# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Evaluation harness for published OpenVLA-7B checkpoints on the LIBERO suites: single-step inference, closed-loop rollouts in the simulator, offline comparison against demos, and plots. It does not train anything. Rollout success rate is the primary metric, and the point of the code is that results stay comparable with the OpenVLA paper.

The target is a Linux server with an RTX 5090 (Blackwell, sm_120). Anything that needs CUDA, EGL, or LIBERO runs only there, not on a macOS checkout. `third_party/`, `data/`, and `outputs/` are gitignored and created on the server.

The README, docstrings, comments, and CLI output are in Vietnamese. Keep new ones in Vietnamese.

## Commands

The package is not installed; run modules from the repo root (default data and output paths are relative to it).

```bash
bash scripts/install_server.sh   # conda env "openvla" (Python 3.10), torch cu128, requirements, LIBERO, then check_env
conda activate openvla
python scripts/check_env.py      # must report [ OK ] for all 4 checks

python -m ovla_eval.download --suites libero_spatial      # checkpoint + RLDS dataset; "all" for every suite
python -m ovla_eval.infer_single --rlds-suite libero_spatial --episode 0 --step 30 --verify --save-input input.png
python -m ovla_eval.rollout_libero --suite libero_spatial --trials 5      # quick; omit --trials for the paper's 50 per task
python -m ovla_eval.offline_eval --suite libero_spatial --max-episodes 20
python -m ovla_eval.plot outputs/offline/libero_spatial
python -m ovla_eval.monitor --rollouts outputs/rollouts/libero_spatial --alarm-signals entropy timer   # numpy only
```

README section 7 has the full failure-monitor and response workflow, including `replay_safety --hold-after` and `compare_responses`.

There are no tests, linter, or build step. The available checks are `check_env.py` for the environment and `infer_single --verify`, which compares `OpenVLAPolicy.predict()` against the upstream `vla.predict_action()`. Run `--verify` after any change to decoding in `policy.py`.

## Architecture

`infer_single`, `offline_eval`, and `rollout_libero` differ only in where the image comes from. All three go through `resize_image` and then `OpenVLAPolicy.predict()`, which returns an `ActionPrediction`. `offline_eval` and `rollout_libero` write one `.npz` per episode plus a `summary.json`, and `plot` reads those files. The `.npz` keys are a contract between the writers, `plot.py`, and the table in the README, so change them together.

`monitor` calibrates runtime failure signals on saved `.npz` files with leave-one-task-out thresholds and compares them with a timer. `ResponseGuard` applies the response during a rollout, and `compare_responses` compares runs by init state. `monitor` needs only numpy, so a macOS checkout can run it.

`OpenVLAPolicy` loads the model from HuggingFace with `trust_remote_code`; the `openvla` package is not installed. It re-implements action decoding rather than calling `predict_action()`, because it also needs token probabilities, entropy over the 256 action tokens, and a soft-decoded expected action. `predict_reference()` keeps the upstream path available as the oracle for `--verify`.

SDPA attention and torch 2.7 cu128 replace upstream's `torch==2.2.0` and `flash-attn==2.5.5`, which have no sm_120 kernels. Do not move back to those pins.

## Invariants that are easy to break

**Paper comparability.** Image preprocessing, gripper post-processing, `NUM_STEPS_WAIT`, `SUITE_MAX_STEPS`, init states, and seed 7 are copied from OpenVLA's `experiments/robot/`. Changing any of them invalidates the comparison with `PAPER_SUCCESS_RATE`.

**Image pipeline.** The right path depends on the source:
- Simulator: `lu.agentview_image(obs)` (rotates 180°), then `resize_image(img)` with the default JPEG roundtrip.
- RLDS: `resize_image(img, jpeg_roundtrip=False)`. These images are already upright and already decoded from JPEG.
- Center crop happens inside `OpenVLAPolicy._prepare_inputs`, so callers must not crop first. Saved frames and videos are therefore uncropped; `infer_single --save-input` is the one output that shows what the model actually sees.

**Gripper conventions.** There are two, and mixing them silently breaks results:
- `raw` / `pred_raw`: continuous in [0, 1], 1 = open (same as `predict_action()`).
- `env` / `pred_env`: binary, -1 = open, +1 = closed (LIBERO). `gt_action` from RLDS uses this convention too, so compare it with `env`, not `raw`.

`gripper_to_env` converts between them. The gripper dimension has `mask=False`, so `unnormalize` and `normalize` leave it unchanged.

**Import order.** Side effects at import time matter:
- `libero_utils` sets `MUJOCO_GL`, `PYOPENGL_PLATFORM`, and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`, so it must be imported before robosuite or mujoco (`rollout_libero` imports it first for this reason). A standalone script must do the same or set these variables itself.
- `preprocess` hides GPUs from TensorFlow, so it must be imported before tensorflow_datasets (`rlds_utils` does this). TF is CPU-only and leaves the GPU to torch.
- `constants` must stay free of heavy imports so `download` works on its own.

**Response runs.** `pred_env` must hold the action actually sent to the env, because `replay_safety` replays that array. With `dz_min`, `token_prob`, `bin_entropy`, and `expected_raw` still come from the unmasked logits, since `out.scores` contains masked logits and the monitor reads these fields. An alarm at step t takes effect at step t + 1 in every mode. Env state carries over between episodes (see `--fresh-env-at`), so response runs cannot reuse baseline episodes: run every episode in order, use a new `--out-dir`, and check `--prefix-check`.

**`unnorm_key`.** It defaults to the suite name, and `resolve_unnorm_key` also tries the `_no_noops` suffix. The base `openvla/openvla-7b` checkpoint has no LIBERO norm stats, so LIBERO evaluation needs the fine-tuned checkpoint for that suite.

**Rollout resume.** Output paths are `<out-dir>/<suite>/task_XX/`, with no checkpoint or settings in the path, and any existing `ep_XXX.npz` is skipped. Running with a different `--checkpoint`, `--unnorm-key`, or `--no-center-crop` in the same `--out-dir` silently mixes results, so use a new `--out-dir`. Episodes that hit a simulator exception count as failures and also leave an `.npz`, so they are not retried; delete the file to rerun one. `summary.json` is overwritten on each run and covers only the tasks passed in that run.

**Version pins.** `transformers`, `tokenizers`, and `timm` are pinned because the HF remote code requires those versions; `check_env.py` hardcodes the same versions, so update both together. torch is pinned in `install_server.sh`, not `requirements.txt`. LIBERO is installed with `--no-deps` because its own requirements would replace the cu128 torch. `mujoco` stays at 3.2.3 because robosuite 1.4.1 is incompatible with newer releases.
