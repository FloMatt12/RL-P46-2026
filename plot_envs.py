# Part 3: learning curves + summary table for a results folder produced by run_envs.py.
#   python plot_envs.py                          # results/part3_envs -> results/part3_envs/learning_curves.png + summary.md
#   python plot_envs.py --tag smoke --no_show
#
# Curves: mean over seeds, shaded +- std_scale * std (paper Fig. 5 uses half a std), optional uniform smoothing.
# Table, per (env, policy), using the same metrics as the paper:
#   max_avg  = max over evaluations of the seed-mean curve, +- std across seeds at that evaluation (Table 1)
#   last10   = mean of the last 10 evaluations, averaged over seeds, +- std across seeds (Table 2)
import argparse
from collections import defaultdict
from math import ceil
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
NAME_RE = re.compile(r"^(?P<policy>[A-Za-z0-9]+)_(?P<env>.+)_(?P<seed>\d+)\.npy$")
COLORS = {"TD3": "C0", "OurDDPG": "C1", "DDPG": "C2"}


def load_runs(results_dir):
	runs = defaultdict(lambda: defaultdict(dict))  # env -> policy -> seed -> curve
	for path in sorted(results_dir.glob("*.npy")):
		match = NAME_RE.match(path.name)
		if match:
			runs[match["env"]][match["policy"]][int(match["seed"])] = np.load(path, allow_pickle=False)
	return runs


def stack_complete(curves_by_seed):
	# Only keep runs of the most common length, so unfinished runs don't distort the mean.
	lengths = [len(c) for c in curves_by_seed.values()]
	n = max(set(lengths), key=lengths.count)
	seeds = sorted(s for s, c in curves_by_seed.items() if len(c) == n)
	return np.stack([curves_by_seed[s] for s in seeds]), seeds


def smooth(y, window):
	if window <= 1:
		return y
	kernel = np.ones(window) / window
	padded = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
	return np.convolve(padded, kernel, mode="valid")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--tag", default="part3_envs")
	parser.add_argument("--eval_freq", type=int, default=5_000)
	parser.add_argument("--std_scale", type=float, default=0.5)
	parser.add_argument("--smooth", type=int, default=1)       # moving-average window (evaluations); 1 = off
	parser.add_argument("--no_show", action="store_true")
	args = parser.parse_args()

	results_dir = ROOT / "results" / args.tag
	runs = load_runs(results_dir)
	if not runs:
		raise SystemExit(f"No result files found in {results_dir}")

	envs = sorted(runs)
	columns = min(4, len(envs))
	rows = ceil(len(envs) / columns)
	fig, axes = plt.subplots(rows, columns, figsize=(4.5 * columns, 3.4 * rows), squeeze=False)

	table = ["| Environment | Policy | Seeds | Steps | Max avg return | Last-10 avg return |", "|---|---|---|---|---|---|"]
	for ax, env in zip(axes.flat, envs):
		for policy in sorted(runs[env], key=lambda p: list(COLORS).index(p) if p in COLORS else 99):
			data, seeds = stack_complete(runs[env][policy])
			steps = np.arange(data.shape[1]) * args.eval_freq
			mean, std = data.mean(axis=0), data.std(axis=0)

			best = int(np.argmax(mean))
			last10 = data[:, -10:].mean(axis=1)
			table.append(
				f"| {env} | {policy} | {len(seeds)} | {steps[-1]:,} | {mean[best]:.1f} ± {std[best]:.1f} "
				f"| {last10.mean():.1f} ± {last10.std():.1f} |"
			)

			m, s = smooth(mean, args.smooth), smooth(std, args.smooth)
			color = COLORS.get(policy)
			ax.plot(steps, m, color=color, label=f"{policy} (n={len(seeds)})")
			ax.fill_between(steps, m - args.std_scale * s, m + args.std_scale * s, color=color, alpha=0.2)
		ax.set_title(env)
		ax.set_xlabel("Time steps")
		ax.set_ylabel("Average return")
		ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
		ax.grid(True, alpha=0.25)
		ax.legend(fontsize=8)

	for ax in axes.flat[len(envs):]:
		ax.set_visible(False)
	fig.tight_layout()

	fig_path = results_dir / "learning_curves.png"
	fig.savefig(fig_path, dpi=200, bbox_inches="tight")
	summary_path = results_dir / "summary.md"
	summary_path.write_text(
		f"Shaded region in {fig_path.name}: ±{args.std_scale} std over seeds.\n\n" + "\n".join(table) + "\n",
		encoding="utf-8",
	)
	print("\n".join(table))
	print(f"\nSaved {fig_path} and {summary_path}")
	if not args.no_show:
		plt.show()


if __name__ == "__main__":
	main()
