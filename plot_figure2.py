# Recreate Figure 2 of paper P46: value overestimation of double-Q actor-critics.
# Run "python overestimation.py" for each policy first (see run_figure2.sh),
# then "python plot_figure2.py --env Hopper-v5" to build the figure from the
# .npz files saved under results/figure2/.

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Display order and labels. DDQN-AC and DQ-AC are the Figure 2 subjects;
# DDPG and CDQ/TD3 bracket them as the uncontrolled and controlled references.
POLICY_ORDER = ["DDPG", "DDQN_AC", "DQ_AC", "TD3"]
POLICY_TITLE = {
    "DDPG": "DDPG (reference)",
    "DDQN_AC": "DDQN-AC (Double DQN)",
    "DQ_AC": "DQ-AC (Double Q-learning)",
    "TD3": "CDQ / TD3 (reference)",
}


def plot_figure2(env="Hopper-v5", results_dir=None, output_path=None, show=True):
    root = Path(results_dir) if results_dir else (
        Path(__file__).resolve().parent / "results" / "figure2"
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Results directory not found: {root}")

    # Collect the policies that actually have data for this env (seed 0).
    found = []
    for policy in POLICY_ORDER:
        matches = sorted(root.glob(f"{policy}_{env}_*.npz"))
        if matches:
            found.append((policy, matches[0]))
    if not found:
        raise FileNotFoundError(f"No results/figure2/*_{env}_*.npz files found in {root}")

    fig, axes = plt.subplots(1, len(found), figsize=(4.2 * len(found), 3.6), squeeze=False)
    for ax, (policy, path) in zip(axes[0], found):
        data = np.load(path, allow_pickle=True)
        steps, est, true = data["steps"], data["estimate"], data["true"]
        ax.plot(steps, est, color="tab:red", label="estimate Q(s,π(s))")
        ax.plot(steps, true, color="tab:blue", linestyle="--", label="true (MC return)")
        ax.fill_between(steps, true, est, where=(est >= true), color="tab:red", alpha=0.12)
        ax.set_title(POLICY_TITLE.get(policy, policy), fontsize=10)
        ax.set_xlabel("training steps")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(5, 5))
        ax.grid(True, alpha=0.3)
    axes[0][0].set_ylabel("average value")
    axes[0][0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Figure 2 reproduction — value overestimation on {env}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if output_path is None:
        output_path = root.parent.parent / f"figure2_{env}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Wrote {output_path}")

    print("\nFinal overestimation bias (last measurement):")
    for policy, path in found:
        data = np.load(path, allow_pickle=True)
        bias = float(data["estimate"][-1] - data["true"][-1])
        print(f"  {POLICY_TITLE.get(policy, policy):28s} bias = {bias:+8.2f}")

    if show:
        plt.show()
    return fig, axes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="Hopper-v5")
    parser.add_argument("--results_dir", default=None)
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()
    plot_figure2(env=args.env, results_dir=args.results_dir, show=not args.no_show)
