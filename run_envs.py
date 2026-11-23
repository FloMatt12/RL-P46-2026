# Part 3 (environment generalisation): runs TD3 and DDPG on environments not used in the paper.
# Each (policy, env, seed) is a separate main.py process; finished runs are skipped on restart.
#
# Examples:
#   python run_envs.py --dry_run                                 # print the job list and exit
#   python run_envs.py --workers 12                              # full default sweep
#   python run_envs.py --envs Pendulum-v1 --seeds 2 --max_timesteps 20000 --tag smoke
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
# start_timesteps follows the repo convention: 25e3 by default, 1e3 for short/simple tasks.
ENVIRONMENTS = {
	# MuJoCo, not used in the paper
	"Swimmer-v5":               (300_000, 25_000),
	"Pusher-v5":                (300_000, 25_000),
	"Humanoid-v5":              (300_000, 25_000),
	"HumanoidStandup-v5":       (300_000, 25_000),
	# Box2D (different physics engine)
	"BipedalWalker-v3":         (300_000, 25_000),
	"LunarLanderContinuous-v3": (300_000, 25_000),
	# Classic control
	"Pendulum-v1":              (100_000, 1_000),
	"MountainCarContinuous-v0": (300_000, 25_000),
}
POLICIES = ["TD3", "OurDDPG"]


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
	] + args.extra


def run_job(name, command, log_path):
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
	parser.add_argument("--seeds", type=int, default=5)             # runs seeds 0..N-1
	parser.add_argument("--first_seed", type=int, default=0)
	parser.add_argument("--max_timesteps", type=int, default=None)  # overrides the per-env budget
	parser.add_argument("--eval_freq", type=int, default=5_000)
	parser.add_argument("--workers", type=int, default=4)
	parser.add_argument("--tag", default="part3_envs")              # results go to results/<tag>/
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

	failures = []
	with ThreadPoolExecutor(max_workers=args.workers) as executor:
		futures = {executor.submit(run_job, name, cmd, log_dir / f"{name}.log"): (name, cmd) for name, cmd in jobs}
		for done, future in enumerate(as_completed(futures), start=1):
			name, cmd = futures[future]
			try:
				seconds = future.result()
			except Exception as exc:
				failures.append(name)
				print(f"FAILED {name}: {exc} (see {log_dir / (name + '.log')})", file=sys.stderr)
				continue
			manifest["runs"][name] = {"command": cmd[2:], "seconds": round(seconds)}
			save_manifest()
			print(f"[{done}/{len(jobs)}] {name} done in {seconds / 60:.1f} min")

	if failures:
		raise SystemExit(f"{len(failures)} run(s) failed: {failures}")


if __name__ == "__main__":
	main()
