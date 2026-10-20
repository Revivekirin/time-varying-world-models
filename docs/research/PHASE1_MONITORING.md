# Phase 1 monitoring schema v1

Training dynamics, update schedule and checkpoint guards are unchanged. Monitoring
is configured separately from algorithm parameters. Existing full-run provenance
still requires exact code hashes: do not replace code in a running process or
bypass provenance to resume an older checkpoint with this revision. Regenerate
collector gates using the established scripts before a new run. Existing preflight
and learner-resume compatibility records are reused where package hashes match.

## Common metrics

Local tracking-ATTEMPT.jsonl and W&B use the same scalar names:

| Namespace | Fields | X axis |
|---|---|---|
| train | episode_return, success (0/1), episode_length, goal_progress, final_goal_distance | step/env |
| eval | success_rate, terminal_success_rate, return_mean, return_std (population), goal_progress_mean | step/env |
| loss | consistency_loss, reward_loss, value_loss, termination_loss, total_loss, pi_loss, grad_norm and other upstream metrics | step/update |
| perf | learner_updates_per_s (sampled update), planner_ms_per_transition | step/env |
| gpu | device_used_mib, device_free_mib, peak_allocated_bytes, peak_reserved_bytes | step/env |
| time | wall_s, environment_recreation_s | step/env |
| system | storage_free_bytes | step/env |

step/eval_env records cumulative validation transitions separately. All losses
remain in existing metrics-*.jsonl per update. Tracking/W&B samples losses every
100 updates (latest sample, not a window mean); episode and evaluation summaries
are emitted every episode/evaluation. Planner milliseconds currently include the
random-action generation interval during random collection, not only learned MPC.

W&B receives a per-seed eval/episodes table and eval/video_0, eval/video_1. Best
checkpoint selection still uses validation success then mean return, never video
appearance. Checkpoint SHA256 values are recorded in summary and local tracking;
large replay/model checkpoint binaries are not uploaded automatically.

## Terminal display

Two tqdm bars show 300,000 environment transitions and 299,999 learner updates,
with resumed initial counters. The 1,000-update initial burst advances the learner
bar even while the environment counter is stationary. Episode return/success and
latest validation success/return appear in the postfix. Validation has its own
seed progress bar; checkpoint/evaluation phases are shown explicitly. A tqdm ETA
is a rate-based estimate, not the enforced wall-time budget.

## Video

Default: first two existing validation seed episodes per evaluation, at 20 FPS,
including reset and terminal frames. No additional rollout. Frame encoding streams
to videos/VALIDATION_REPORT_STEM/seed-SEED.mp4. Uses the pinned native
render_rgb_array method and imageio/libx264. State observations remain unchanged;
training does not render. RNG is saved/restored around rendering and W&B logging.
Rendering/encoding overhead counts toward the existing deadlines. GPU video
runtime is UNRESOLVED locally; the existing collector gate's frozen evaluator now
exercises this path. Missing rendering/codec support fails visibly, not with a
fabricated successful video. --video-episodes 0, 1 or 2 controls the bounded count.

## W&B configuration and execution

Dependencies: tqdm, wandb, imageio, imageio-ffmpeg plus a working ManiSkill renderer.
No lock files were changed and no packages installed by the assistant. Check the
server packages before running; if missing, resolve optional packages without
silently upgrading pinned dependencies. Package drift still fails existing gates.

```bash
python -c "import wandb, tqdm, imageio, imageio_ffmpeg; print('monitoring imports OK')"
wandb login
export WANDB_MODE=online
export WANDB_PROJECT=tvwm-phase1
# Optional team workspace:
# export WANDB_ENTITY=your-team
bash scripts/phase1_integrated_launch.sh
```

The integrated script includes BOTH collector gates and approved full training;
use only after syncing files and with new/unused output paths. Existing paths are
never overwritten. If a previous launch is running, leave its code/process intact;
this instrumentation is not injected into it automatically.

CLI equivalents: --wandb-mode online --wandb-project tvwm-phase1
--wandb-entity TEAM --video-episodes 2. Default W&B mode is offline, project
 tvwm-phase1. Offline runs retain uploadable W&B files locally; wandb sync can upload
the generated offline-run directory later. --wandb-mode disabled retains the local
tracking schema, video and tqdm without importing wandb.

Each execution attempt gets a separate W&B run, grouped by task and training seed.
On checkpoint rollback this prevents mixing repeated environment steps with a
previous attempt's history. Restored counters, attempt ID, full provenance and
configuration are attached; wandb-ATTEMPT.json records the run ID/URL. There is no
implicit W&B resume or assumption that training counters are globally monotonic
across attempts. observability-ATTEMPT.json records mode, package versions and
video/logging configuration. W&B/API/auth failures are explicit; no silent switch
from online to offline and no hyperparameter changes.

API references: [W&B Run / define_metric](https://docs.wandb.ai/ref/python/experiments/run/)
and [official Video implementation](https://github.com/wandb/wandb/blob/main/wandb/sdk/data_types/video.py).

Local validation: 46 tests discovered, 30 passed, 16 dependency skips. Remote W&B
upload and GPU video encoding remain UNRESOLVED until the server workflow runs.
