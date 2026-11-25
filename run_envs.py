# Part 3 (environment generalisation): runs TD3 and DDPG on environments not used in the paper.
# Each (policy, env, seed) is a separate main.py process; finished runs are skipped on restart.
#
# Examples:
#   python run_envs.py --dry_run                                 # print the job list and exit
#   python run_envs.py --workers 16 --tag paper_1M               # full sweep: paper hyperparameters, 1M steps, 10 seeds
#   python run_envs.py --envs Pendulum-v1 --seeds 2 --max_timesteps 20000 --tag smoke
#   python run_envs.py --envs MountainCarContinuous-v0 --tag mcar_noise05 -- --expl_noise 0.5   # extra args go to main.py
# The first (CPU) sweep used: --hparams repo --max_timesteps 300000 --start_timesteps 25000 --seeds 3 (Pendulum: 100k, 1k).
#
# Output (results/<tag>/):
#   {policy}_{env}_{seed}.npy   evaluations, same format as main.py (index 0 = untrained policy)
#   logs/{policy}_{env}_{seed}.log
#   manifest.json               commands, git commit, library versions, machine info
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent

# Environments not in the paper. Value = (max_timesteps, start_timesteps).
# Paper protocol: 1M steps, 1k random start steps (the paper used 10k only for HalfCheetah and Ant).
# Pendulum converges in ~10k steps, so it is kept as a short sanity check.
ENVIRONMENTS = {
	# MuJoCo, not used in the paper
	"Swimmer-v5":               (1_000_000, 1_000),
	"Pusher-v5":                (1_000_000, 1_000),
	"Humanoid-v5":              (1_000_000, 1_000),
	"HumanoidStandup-v5":       (1_000_000, 1_000),
	# Box2D (different physics engine)
	"BipedalWalker-v3":         (1_000_000, 1_000),
	"LunarLanderContinuous-v3": (1_000_000, 1_000),
	# Classic control
	"Pendulum-v1":              (100_000, 1_000),
	"MountainCarContinuous-v0": (1_000_000, 1_000),
}
POLICIES = ["TD3", "OurDDPG"]

# Network/optimiser settings passed to main.py.
#   paper: Fujimoto et al. 2018, Sec. 6.1 (400-300 networks, Adam lr 1e-3, batch 100), same for TD3 and OurDDPG
#   repo:  current sfujim/TD3 defaults (256-256, lr 3e-4, batch 256), i.e. main.py's own defaults
HPARAMS = {
	"paper": ["--hidden_sizes", "400", "300", "--actor_lr", "1e-3", "--critic_lr", "1e-3", "--batch_size", "100"],
	"repo": [],
}


def expected_evals(max_timesteps, eval_freq):
	return 1 + max_timesteps // eval_freq


def result_is_complete(path, n_expected):
	if not path.exists():
		return False
	try:
		evaluations = np.load(path, allow_pickle=False)
	except (OSError, ValueError, EOFError):
		return False
	return evaluations.ndim == 1 and len(evaluations) == n_expected


def build_command(policy, env, seed, max_timesteps, start_timesteps, args, out_dir):
	return [
		sys.executable, "-u", str(ROOT / "main.py"),
		"--policy", policy,
		"--env", env,
		"--seed", str(seed),
		"--max_timesteps", str(max_timesteps),
		"--start_timesteps", str(start_timesteps),
		"--eval_freq", str(args.eval_freq),
		"--results_dir", str(out_dir),
	] + HPARAMS[args.hparams] + args.extra


class NotStarted(Exception):
	pass


def check_envs(envs):
	# Fail in seconds, not after hours of other runs, if an env can't be built (e.g. a missing package).
	import gymnasium as gym
	broken = {}
	for env in envs:
		try:
			gym.make(env).close()
		except Exception as exc:
			broken[env] = f"{type(exc).__name__}: {exc}"
	return broken


def log_tail(log_path, n=15):
	try:
		return "".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-n:])
	except OSError:
		return ""


def run_job(name, command, log_path, start_deadline=None):
	if start_deadline is not None and time.time() > start_deadline:
		raise NotStarted(name)
	child_env = os.environ.copy()
	for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
		child_env[var] = "1"
	start = time.time()
	with log_path.open("a", encoding="utf-8", buffering=1) as log:
		log.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] {' '.join(command)}\n")
		subprocess.run(command, cwd=ROOT, check=True, stdout=log, stderr=subprocess.STDOUT, env=child_env)
	return time.time() - start


def library_versions():
	versions = {"python": sys.version.split()[0]}
	for module in ("numpy", "torch", "gymnasium", "mujoco", "Box2D"):
		try:
			versions[module] = __import__(module).__version__
		except Exception:
			versions[module] = None
	return versions


def git_commit():
	try:
		commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
		dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
		return commit + ("-dirty" if dirty else "")
	except OSError:
		return None


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--policies", nargs="+", default=POLICIES)
	parser.add_argument("--envs", nargs="+", default=list(ENVIRONMENTS))
	parser.add_argument("--seeds", type=int, default=10)            # runs seeds first_seed..first_seed+N-1 (paper: 10)
	parser.add_argument("--first_seed", type=int, default=0)
	parser.add_argument("--hparams", choices=list(HPARAMS), default="paper")
	parser.add_argument("--max_timesteps", type=int, default=None)  # overrides the per-env budget
	parser.add_argument("--start_timesteps", type=int, default=None)  # overrides the per-env random start steps
	parser.add_argument("--eval_freq", type=int, default=5_000)
	parser.add_argument("--workers", type=int, default=4)
	parser.add_argument("--tag", default="part3_envs")              # results go to results/<tag>/
	parser.add_argument("--no_start_after_hours", type=float, default=None)  # don't start new runs after this (job time limit)
	parser.add_argument("--dry_run", action="store_true")
	parser.add_argument("extra", nargs=argparse.REMAINDER)          # after "--": passed through to main.py
	args = parser.parse_args()
	if args.extra and args.extra[0] == "--":
		args.extra = args.extra[1:]

	unknown = [e for e in args.envs if e not in ENVIRONMENTS]
	if unknown:
		parser.error(f"unknown env(s) {unknown}; add them to ENVIRONMENTS with a budget")

	out_dir = ROOT / "results" / args.tag
	log_dir = out_dir / "logs"

	jobs, skipped = [], 0
	for seed in range(args.first_seed, args.first_seed + args.seeds):
		for env in args.envs:
			max_t, start_t = ENVIRONMENTS[env]
			if args.max_timesteps is not None:
				max_t = args.max_timesteps
			if args.start_timesteps is not None:
				start_t = args.start_timesteps
			for policy in args.policies:
				name = f"{policy}_{env}_{seed}"
				if result_is_complete(out_dir / f"{name}.npy", expected_evals(max_t, args.eval_freq)):
					skipped += 1
					continue
				jobs.append((name, build_command(policy, env, seed, max_t, start_t, args, out_dir)))

	total_steps = sum(int(cmd[cmd.index("--max_timesteps") + 1]) for _, cmd in jobs)
	print(f"Skipping {skipped} complete run(s); scheduling {len(jobs)} ({total_steps / 1e6:.1f}M env steps).")
	if args.dry_run:
		for name, cmd in jobs:
			print(" ", name, " ".join(cmd[3:]))
		return

	broken = check_envs(sorted({cmd[cmd.index("--env") + 1] for _, cmd in jobs}))
	if broken:
		raise SystemExit("Cannot create environment(s), nothing started:\n"
			+ "\n".join(f"  {env}: {err}" for env, err in broken.items()))

	log_dir.mkdir(parents=True, exist_ok=True)
	manifest_path = out_dir / "manifest.json"
	manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"launches": [], "runs": {}}
	manifest["launches"].append({
		"time": datetime.now().isoformat(timespec="seconds"),
		"argv": sys.argv,
		"git_commit": git_commit(),
		"versions": library_versions(),
		"machine": {"platform": platform.platform(), "processor": platform.processor(), "cpus": os.cpu_count()},
	})

	def save_manifest():
		manifest_path.write_text(json.dumps(manifest, indent=2))

	save_manifest()

	start_deadline = None
	if args.no_start_after_hours is not None:
		start_deadline = time.time() + args.no_start_after_hours * 3600

	failures, not_started = [], []
	with ThreadPoolExecutor(max_workers=args.workers) as executor:
		futures = {executor.submit(run_job, name, cmd, log_dir / f"{name}.log", start_deadline): (name, cmd)
			for name, cmd in jobs}
		for done, future in enumerate(as_completed(futures), start=1):
			name, cmd = futures[future]
			try:
				seconds = future.result()
			except NotStarted:
				not_started.append(name)
				continue
			except Exception as exc:
				failures.append(name)
				log_path = log_dir / f"{name}.log"
				print(f"FAILED {name}: {exc}\n--- last lines of {log_path} ---\n{log_tail(log_path)}", file=sys.stderr)
				continue
			manifest["runs"][name] = {"command": cmd[2:], "seconds": round(seconds)}
			save_manifest()
			print(f"[{done}/{len(jobs)}] {name} done in {seconds / 60:.1f} min")

	if not_started:
		print(f"Not started ({args.no_start_after_hours} h limit reached), run again to do them: {not_started}")
	if failures:
		raise SystemExit(f"{len(failures)} run(s) failed: {failures}")


if __name__ == "__main__":
	main()
