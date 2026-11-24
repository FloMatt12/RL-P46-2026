# Reproduce the overestimation-bias measurement from the TD3 paper (Figure 2 /
# Section 4). Trains one policy and periodically compares the critic's value
# estimate against the true discounted return of the current policy.
#
# For a chosen state s:
#   estimate(s) = Q(s, pi(s))                     # what the critic predicts
#   true(s)     = sum_t gamma^t r_t  following pi # Monte-Carlo rollout return
#   bias        = estimate - true                 # > 0 means overestimation
#
# Estimate and true value are both measured on the states the policy actually
# visits (on-policy rollouts), so no simulator-state snapshots are needed.
#
# Run one policy:
#   python overestimation.py --policy DDPG   --env Hopper-v5 --max_timesteps 200000
#   python overestimation.py --policy DDQN_AC --env Hopper-v5 --max_timesteps 200000
#   python overestimation.py --policy DQ_AC  --env Hopper-v5 --max_timesteps 200000
#   python overestimation.py --policy TD3    --env Hopper-v5 --max_timesteps 200000
# Then build the figure with:  python plot_figure2.py --env Hopper-v5

import argparse
import os

import numpy as np
import gymnasium as gym
import torch

import utils
import TD3
import OurDDPG
import DDPG
import DDQN_AC
import DQ_AC


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_policy(name, state_dim, action_dim, max_action, args):
    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }
    if name == "TD3":
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        return TD3.TD3(**kwargs)
    if name == "DDQN_AC":
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        return DDQN_AC.DDQN_AC(**kwargs)
    if name == "DQ_AC":
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        return DQ_AC.DQ_AC(**kwargs)
    if name == "OurDDPG":
        return OurDDPG.DDPG(**kwargs)
    if name == "DDPG":
        return DDPG.DDPG(**kwargs)
    raise ValueError(f"unknown policy {name!r}")


def value_estimate(policy, states):
    # Average critic value Q(s, pi(s)) over a batch of states, dispatched by the
    # critic interface each policy exposes.
    state_tensor = torch.FloatTensor(np.asarray(states)).to(device)
    with torch.no_grad():
        if hasattr(policy, "current_value"):          # DDQN_AC, DQ_AC
            values = policy.current_value(state_tensor)
        else:
            action = policy.actor(state_tensor)
            out = policy.critic(state_tensor, action)
            if isinstance(out, tuple):                # TD3 twin critic -> (q1, q2)
                values = 0.5 * (out[0] + out[1])
            else:                                      # DDPG / OurDDPG single critic
                values = out
    return values.cpu().numpy().reshape(-1)


def measure_bias(policy, env_name, gamma, episodes, seed):
    # Roll out the deterministic policy, then compare the critic estimate to the
    # discounted Monte-Carlo return from every visited state.
    eval_env = gym.make(env_name)
    states = []
    true_returns = []
    for ep in range(episodes):
        state, _ = eval_env.reset(seed=seed + ep)
        ep_states = []
        ep_rewards = []
        while True:
            action = policy.select_action(np.array(state))
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            ep_states.append(np.array(state, dtype=np.float32))
            ep_rewards.append(reward)
            state = next_state
            if terminated or truncated:
                break
        # Discounted return-to-go within the episode.
        g = 0.0
        returns = np.zeros(len(ep_rewards), dtype=np.float64)
        for t in range(len(ep_rewards) - 1, -1, -1):
            g = ep_rewards[t] + gamma * g
            returns[t] = g
        states.extend(ep_states)
        true_returns.extend(returns.tolist())
    eval_env.close()

    estimates = value_estimate(policy, states)
    mean_est = float(np.mean(estimates))
    mean_true = float(np.mean(true_returns))
    return mean_est, mean_true


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                    # TD3, DDPG, OurDDPG, DDQN_AC, DQ_AC
    parser.add_argument("--env", default="Hopper-v5")                 # Gymnasium MuJoCo env
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start_timesteps", default=10e3, type=int)  # Initial random policy steps
    parser.add_argument("--max_timesteps", default=200e3, type=int)   # Training steps
    parser.add_argument("--measure_freq", default=10e3, type=int)     # How often to measure bias
    parser.add_argument("--measure_episodes", default=5, type=int)    # Rollouts per measurement
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--tau", default=0.005, type=float)
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--noise_clip", default=0.5, type=float)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--torch_threads", default=1, type=int)
    args = parser.parse_args()

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Overestimation diagnostic | Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    os.makedirs("./results/figure2", exist_ok=True)

    env = gym.make(args.env)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    policy = make_policy(args.policy, state_dim, action_dim, max_action, args)
    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    steps_log, est_log, true_log = [], [], []

    state, _ = env.reset(seed=args.seed)
    for t in range(int(args.max_timesteps)):
        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done_bool = float(terminated)
        replay_buffer.add(state, action, next_state, reward, done_bool)
        state = next_state

        if terminated or truncated:
            state, _ = env.reset()

        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if (t + 1) % args.measure_freq == 0 and t >= args.start_timesteps:
            mean_est, mean_true = measure_bias(
                policy, args.env, args.discount, args.measure_episodes, args.seed + t
            )
            steps_log.append(t + 1)
            est_log.append(mean_est)
            true_log.append(mean_true)
            print(
                f"T: {t+1:>7d}  estimate={mean_est:8.2f}  true={mean_true:8.2f}  "
                f"bias={mean_est - mean_true:+8.2f}",
                flush=True,
            )
            np.savez(
                f"./results/figure2/{file_name}.npz",
                steps=np.array(steps_log),
                estimate=np.array(est_log),
                true=np.array(true_log),
                policy=args.policy,
                env=args.env,
                seed=args.seed,
            )
    env.close()
    print(f"Saved ./results/figure2/{file_name}.npz")
