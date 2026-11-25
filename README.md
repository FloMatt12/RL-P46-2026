# Addressing Function Approximation Error in Actor-Critic Methods

### Setup
Install python \
Create an environmnets and activate it
```
py -m venv .venv
.venv\Scripts\activate
```
\
Install all dependencies (gymnasium[mujoco], numpy, torch, matplotlib)
```
pip install gymnasium[mujoco] numpy torch matplotlib
```
Run a .sh file:
1. open git bash
2. navigate to the folder of the programming environment (u can go back with cd .. all the way to /d or /c)
3. run: source .venv/Scripts/activate
4. run: ./run_experiments.sh
### Usage
The paper results can be reproduced by running:
```
./run_experiments.sh
```
Experiments on single environments can be run by calling:
```
python main.py --env HalfCheetah-v5
```

Hyper-parameters can be modified with different arguments to main.py. We include an implementation of DDPG (DDPG.py), which is not used in the paper, for easy comparison of hyper-parameters with TD3. This is not the implementation of "Our DDPG" as used in the paper (see OurDDPG.py). 

### Figure 2 — Overestimation bias

Reproduces the overestimation-bias experiment from Section 4 of the paper
(Figure 2): the critic's value estimate `Q(s, π(s))` is compared against the
true discounted return of the current policy, measured throughout training on
a MuJoCo task. Two extra actor-critic baselines are added for this study:

- `DDQN_AC.py` — Double-DQN actor-critic: the online actor selects the next action, the target critic evaluates it (single critic, no clipped min).
- `DQ_AC.py` — Double Q-learning actor-critic: two independent actor-critic pairs with cross-evaluated targets.

Both share TD3's architecture, target smoothing and delayed updates, so the
only difference from CDQ/TD3 is the target-value formula — isolating the effect
of the double-Q estimator, as in the paper.

Reproduce the figure (DDPG, DDQN-AC, DQ-AC, CDQ/TD3 on Hopper-v5, 200k steps):
```
./run_figure2.sh
```
Or a single method / environment:
```
python overestimation.py --policy DQ_AC --env Hopper-v5 --max_timesteps 200000
python plot_figure2.py --env Hopper-v5
```
Per-method data is written to `results/figure2/*.npz`; the combined figure is
saved to `figure2_<env>.png`. Expected result: DDPG and DDQN-AC overestimate
strongly (estimate well above the true return), DQ-AC much less, and CDQ/TD3
stays close to (or slightly below) the true value.

Algorithms which TD3 compares against (PPO, TRPO, ACKTR, DDPG) can be found at [OpenAI baselines repository](https://github.com/openai/baselines). 

### Results
Code is no longer exactly representative of the code used in the paper. Minor adjustments to hyperparamters, etc, to improve performance. Learning curves are still the original results found in the paper.

Learning curves found in the paper are found under /learning_curves. Each learning curve are formatted as NumPy arrays of 201 evaluations (201,), where each evaluation corresponds to the average total reward from running the policy for 10 episodes with no exploration. The first evaluation is the randomly initialized policy network (unused in the paper). Evaluations are peformed every 5000 time steps, over a total of 1 million time steps. 

Numerical results can be found in the paper, or from the learning curves. Video of the learned agent can be found [here](https://youtu.be/x33Vw-6vzso). 



### Part 3: other environments
Tests whether TD3's advantage over DDPG ("OurDDPG", the paper's re-tuned DDPG) holds on environments not used in the paper: MuJoCo (Swimmer, Pusher, Humanoid, HumanoidStandup), Box2D (BipedalWalker, LunarLanderContinuous) and classic control (Pendulum, MountainCarContinuous).

The Box2D environments need extra dependencies:
```
pip install swig
pip install "gymnasium[box2d]"
```
Or install the exact versions used with `conda env create -f environment.yml` (this file is also what RenkuLab builds from).

Run the sweep and plot it:
```
python run_envs.py --dry_run                      # list the jobs
python run_envs.py --workers 16 --tag paper_1M    # TD3 + OurDDPG, 10 seeds, 1M steps (Pendulum 100k), paper hyperparameters
python plot_envs.py --tag paper_1M                # results/paper_1M/learning_curves.png + summary.md
```

On RenkuLab, run it as a job with `renku_job.sh`. A job's files are deleted when it ends, so the script pushes the results every 30 minutes to a git repo (here `rl-p46-results`, added to the Renku project as a second code repository; a branch name instead pushes to that branch of this repo) and resumes from there if restarted:
`["bash", "RL-P46-2026/renku_job.sh", "rl-p46-results", "--workers", "14", "--tag", "paper_1M"]`

By default the sweep follows the paper's protocol: 400-300 networks, Adam lr 1e-3, batch 100 and 1k random start steps (`--hparams paper`). `--hparams repo` uses the repo defaults instead (256-256, lr 3e-4, batch 256). `--seeds`, `--first_seed`, `--max_timesteps` and `--start_timesteps` override the defaults, so seeds can be added later, e.g. `--first_seed 5 --seeds 5`.

Finished runs are skipped when the sweep is restarted. The results, logs and `manifest.json` (commands, git commit, library versions) are written to `results/<tag>/`. Arguments after `--` are passed through to `main.py`, for example `python run_envs.py --envs MountainCarContinuous-v0 --tag mcar_noise05 -- --expl_noise 0.5`.

New `main.py` options (the defaults keep the previous behaviour): `--hidden_sizes H1 H2`, `--actor_lr`, `--critic_lr` (TD3 and OurDDPG only) and `--results_dir`. `--policy_noise` and `--noise_clip` are now parsed as floats.
