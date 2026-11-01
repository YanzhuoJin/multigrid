# SocialRL: Independent PPO on MultiGrid

This project completes the SocialRL coursework with a single-agent PPO example,
three independent PPO agents in `MultiGrid-Cluttered-Fixed-15x15`, reproducible
training, evaluation against a random policy, learning curves, and annotated
trajectory videos. `REPORT.md` explains the implementation and reports the
measured results from the included runs.

The assignment training target is **100,000 completed joint episodes**. A joint
episode contains all three agents and lasts 100 environment steps in this task.
The episode counter does not count individual agents as separate episodes.

## Included measured results

The included run completed **100,000 joint episodes**, **10,000,000 environment
transitions**, and **30,000,000 agent decisions** in 13.29 minutes. On 200 held-out
episode seeds per policy, sampled IPPO obtained mean collective return **8.86236**
versus **0.05733** for uniform-random actions. All three agents reached the goal
at least once in **96.5%** of evaluation episodes. These results come from one
training seed on one fixed map. See `REPORT.md` for the protocol and limitations.

Validation: **23 automated tests passed** in the installed project environment,
and **55 final artifact checks passed**. `VALIDATION.json` and
`results/ippo_seed1/submission_audit.json` retain the checks. The W&B offline
run has been synchronized and independently verified: all 1,000 logged curve
points, through episode 100,000, match the local 100-episode return means.
[View the W&B run](https://forge.coreweave.com/wandb/311yanzhuo-jin299-medical-college-of-wisconsin/socialrl-ippo/runs/nksxtadt).
The verification record is `results/ippo_seed1/wandb_sync_status.json`.
Creating a remote GitHub fork has not been performed.
For a concise Chinese guide, read `RUN_GUIDE_zh.md`.

## Installation

Use **Python 3.10–3.12**. The project was verified on macOS Apple Silicon with
Python 3.12 and CPU PyTorch. CUDA is optional; the default uses one CPU thread.

Run commands from this project directory. On macOS or Linux:

```bash
bash setup.sh
source ../.venv-socialrl/bin/activate
```

`setup.sh` creates the virtual environment, installs `requirements.txt`, and
runs the tests. To choose a particular interpreter or environment location:

```bash
PYTHON_BIN=python3.12 VENV_DIR=../.venv-socialrl bash setup.sh
```

Windows PowerShell:

```powershell
py -3.12 -m venv ..\.venv-socialrl
$socialrlPython = (Resolve-Path ..\.venv-socialrl\Scripts\python.exe).Path
& $socialrlPython -m pip install --upgrade pip
& $socialrlPython -m pip install -r requirements.txt
& $socialrlPython -m pytest -q
```

For the commands below, Windows users can replace `python` with
`& $socialrlPython`. The shell setup script is for macOS/Linux; the PowerShell
commands perform the corresponding installation on Windows.

The bundled environment requires the legacy packages `gym==0.23.1` and
`gym-minigrid==1.0.3`, together with `numpy==1.26.4`. Installing the separate
modern `minigrid` package does not replace these dependencies. Gym prints a
maintenance notice on import; the bundled compatibility fixes and regression
tests cover the APIs used here. `requirements-lock-macos-py312.txt` records the
versions installed for the supplied experiment.

## Check the environment manually

```bash
python manual_control_multigrid.py --env_name MultiGrid-Cluttered-Fixed-15x15 --seed 52
```

Enter one comma-separated action per agent, for example `2,2,2`. Actions are
`0=left`, `1=right`, `2=forward`, `3=pickup`, `4=drop`, `5=toggle`, and `6=done`.
The `done` action is a no-op, not an instruction to end the episode. Enter `r`
to reset or `q` to quit. The terminal prints the map, individual rewards, and
cumulative collective reward; a local graphical session also displays the map.

For an SSH session without a display, select Matplotlib's noninteractive
backend. This action-and-exit check was run successfully:

```bash
printf '2,2,2\nq\n' | MPLBACKEND=Agg python manual_control_multigrid.py --seed 52 --multiple_episodes 1
```

For interactive text control over SSH, use
`MPLBACKEND=Agg python manual_control_multigrid.py`. A warning that the Agg
canvas cannot be shown is expected; the printed map and keyboard input still
work. The Windows PowerShell equivalent of the scripted check is:

```powershell
$env:MPLBACKEND = 'Agg'
@('2,2,2', 'q') | & $socialrlPython manual_control_multigrid.py --seed 52 --multiple_episodes 1
```

Training, evaluation, and file-based video rendering do not need a GUI.

## Train single-agent PPO

```bash
python ppo.py
```

`ppo.py` is a thin entry point into the same tested PPO implementation. It
defaults to 2,000 episodes in
`MultiGrid-Cluttered-Fixed-Single-6x6-v0`, with outputs in
`results/single_agent`. The command for the supplied single-agent run is:

```bash
python ppo.py --total-episodes 2000 --num-envs 16 --num-steps 128 --seed 1 --log-every 200 --checkpoint-every 1000 --wandb-mode disabled --run-dir results/single_agent_seed1
```

This small environment is a functionality and learning check, not a direct
performance comparison with the larger three-agent task. In the inherited
single-agent environment, `n_agents=1` enables competitive termination: reaching
the goal ends the episode. Consequently, its episode lengths can be shorter
than its time limit. The earlier supplied PPO script is preserved in
`legacy/ppo_original.py` for reference and is not the active training entry point.

## Train independent PPO for the assignment

```bash
python main.py
```

The default configuration in `config/mode/ppo.yaml` requests exactly 100,000
completed episodes, with 16 vectorized environments, a rollout length of 128,
and offline W&B logging. The supplied full experiment uses 64 environments for
higher throughput. Other starter configuration files are retained for reference;
the current CLI loads `config/mode/ppo.yaml` only unless `--config` is provided:

```bash
python main.py --total-episodes 100000 --num-envs 64 --num-steps 128 --seed 1 --wandb-mode offline --wandb-project socialrl-ippo --run-dir results/ippo_seed1
```

Use a new `--run-dir` for a new experiment. Training refuses to overwrite an
existing episode log. A short verification run can use:

```bash
python main.py --total-episodes 17 --num-envs 8 --num-steps 64 --log-every 8 --checkpoint-every 8 --wandb-mode disabled --run-dir results/smoke_new
```

Every agent has its own actor-critic network, Adam optimizer, rollout buffer,
rewards, advantages, and value targets. A policy receives only its own 5×5 local
image and heading. Image channels represent object/color/state categories, so
they are one-hot encoded rather than treated as RGB values. Each independent
network has two 64-unit Tanh hidden layers and separate policy/value heads.
The encoder is shared between the actor and critic *within one agent*; no
parameters are shared between agents. Batched rollout inference stacks copies
of the independent weights for speed without mixing agents or their data.

PPO uses a clipped policy objective, clipped value loss, entropy regularization,
GAE, gradient clipping, and learning-rate annealing. Collective reward is the
sum of the agents' undiscounted rewards in one joint episode and is used for
reporting only. It is never substituted for an agent's own training reward.

## Environment and episode semantics

The three-agent assignment environment has a fixed 15×15 layout, 30 obstacles,
a goal, 5×5 local views, and a 100-step time limit. Layout seed 52 is shared
across vectorized instances; each instance has a separate seeded stream for
agent placement and action order. Wall-clock reseeding has been removed.

In this noncompetitive environment, reaching the goal gives that agent the
original reward `1 - 0.9 * step_count / max_steps` and respawns it. The same
agent can earn more than one goal reward during the joint episode. Its sticky
`env.done[i]` flag records whether it has succeeded at least once; it does not
mean that its later actions should be ignored. Goal respawns continue the
agent's trajectory.

The trainer treats the 100-step limit as a truncation: it bootstraps from the
actual final observation, then cuts the GAE trace before the reset observation.
True goal termination in the competitive single-agent task has no bootstrap.
This explicitly uses time-limit bootstrapping rather than treating every
timeout as a terminal state. An exact observation cache accelerates the fixed
layout while retaining visible moving agents; cached and uncached observations
are checked for equality by the environment tests.

## Checkpoints and resuming

```bash
python main.py --resume results/ippo_seed1/checkpoints/latest.pt
```

The checkpoint supplies the original configuration. It contains all independent
model and optimizer states, environment states, unfinished episodes, counters,
and random-number-generator states. Checkpoints are written atomically after
completed PPO updates. Because saving occurs at a rollout boundary, a periodic
checkpoint may contain slightly more than its episode interval. The final
checkpoint contains the exact requested episode count.

After an interruption, resuming trims CSV and update-log rows newer than the
saved checkpoint and continues from that saved state. The tests verify that an
interrupted/resumed CPU run produces exactly the same model weights as an
uninterrupted run. To extend a completed run, provide a larger target, for
example `--total-episodes 120000`; this changes the learning-rate schedule for
the continuation. The algorithm cannot recover work performed after the most
recent saved checkpoint.

## Weights & Biases

Offline mode saves a real W&B run under the experiment's `wandb/` directory and
does not require an account or network connection. Every completed episode is
also saved in `episodes.csv`. W&B's `train/collective_return` is the mean over
the latest logging window, normally 100 episodes, with `episode` as its x-axis.
`train/collective_return_last` is the last individual episode in that window.
Individual agent returns and success rates are logged separately.

The included run is already uploaded. To authenticate and synchronize another
offline run, or repeat synchronization, use:

```bash
python scripts/sync_wandb.py
```

W&B credentials belong to your W&B account; an OpenAI API key cannot replace them.
The helper passes existing system proxy settings to W&B's background process
when no explicit proxy environment variables are set; it does not change OS settings.
Alternatively, use the W&B CLI directly:

```bash
wandb login
wandb sync results/ippo_seed1/wandb/offline-run-<actual-run-directory>
```

Replace the placeholder with the directory actually present in `wandb/`. To
stream a new run directly, use `--wandb-mode online` and a new run directory
after `wandb login`. Browser login alone may not authenticate the Python SDK.
Use `--wandb-mode disabled` to disable W&B while retaining CSV/checkpoint
outputs. W&B starts a local logging service even in offline mode; restricted
execution sandboxes must permit its local socket.

## Evaluate and export results

```bash
python evaluate.py --checkpoint results/ippo_seed1/checkpoints/final.pt --episodes 200 --seed 1000000 --output results/ippo_seed1/evaluation
python scripts/plot_results.py --run-dir results/ippo_seed1 --window 1000 --target-episodes 100000
```

Evaluation compares the stochastic trained policy, greedy trained policy, and
uniform-random actions on the same 200 held-out episode seeds. The obstacle
layout remains the training layout; this tests new starting configurations and
randomness, not unseen-layout generalization. No learning occurs during
evaluation. Reported intervals describe variability across evaluation episodes
from one trained checkpoint, not uncertainty across independent training seeds.

The first evaluation episode is selected in advance for both trained-policy
videos. Each frame shows the full environment, each agent's local view, its
next action, and accumulated rewards. Selection does not filter for successful
episodes. The exporter produces MP4/GIF videos, a final PNG, and a JSON
trajectory. Add `--no-video` for a faster metrics-only evaluation.

The plotting script reads recorded episodes without extrapolation and exports
PNG, PDF, and SVG figures with source/format notes. It labels a run as partial
until the requested number of episodes is present. For the single-agent run:

```bash
python evaluate.py --checkpoint results/single_agent_seed1/checkpoints/final.pt --episodes 100 --seed 1000000 --output results/single_agent_seed1/evaluation --no-video
python scripts/plot_results.py --run-dir results/single_agent_seed1 --window 200 --target-episodes 2000
```

## Deliverable files

The completed project is located at
`/Users/yanzhuojin/Desktop/socialrl/multigrid`; the submission archive is
`/Users/yanzhuojin/Desktop/socialrl/socialrl_submission.zip`.

```text
REPORT.md                              Method, measured results, and limitations
main.py                                Three-agent IPPO training entry point
ppo.py                                 Single-agent PPO entry point
ippo_agent.py                          Policy, separate buffers, GAE, PPO updates
multiagent_metacontroller.py            Rollouts, logging, checkpoint/resume
env_adapter.py                         Seeded legacy environment factory
config/mode/ppo.yaml                    Assignment training defaults
evaluate.py                            Held-out comparison and annotated videos
scripts/plot_results.py                 Curves and evaluation plots
tests/                                 Environment, algorithm, and output checks
results/ippo_seed1/
  config.json                          Exact configuration used for the run
  episodes.csv                         Every completed training episode
  updates.jsonl                        Per-agent PPO update diagnostics
  summary.json                         Completion counters and runtime
  checkpoints/final.pt                 Final independent policies and runtime
  checkpoints/latest.pt                Most recent resumable checkpoint
  wandb/                               Offline W&B training data
  evaluation/summary.json               Held-out numerical results
  evaluation/episodes.csv               Per-episode evaluation measurements
  evaluation/trajectory_*.mp4           Annotated trained-policy trajectories
  evaluation/trajectory_*.gif           Browser-friendly trajectory animations
  figures/learning_curve.*              Collective return versus episode
  figures/evaluation_comparison.*       Trained policies versus random actions
results/single_agent_seed1/             Corresponding single-agent evidence
```

Use `summary.json` and `REPORT.md` for the measured completion status and final
numbers. Configuration targets alone do not establish that training completed.

## Validation and attribution

```bash
python -m pytest -q
```

Tests cover deterministic environment replay, checkpoint portability, exact
cached observation equivalence, reward/respawn semantics, timeout GAE,
observation privacy, independent parameter and buffer storage, batched inference
equivalence, exact episode counting, resumable training, and evaluation/plot
utilities. See `THIRD_PARTY_NOTICES.md` and `LICENSE` for MultiGrid/MiniGrid and
the supplied CleanRL-style PPO attribution.
