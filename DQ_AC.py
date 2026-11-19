import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Double Q-learning actor-critic (DQ-AC) baseline used in the overestimation
# study. Paper: Fujimoto et al. (2018), Section 4 "Overestimation Bias".
#
# Actor-critic analogue of Double Q-learning (van Hasselt, 2010): two
# independent actor-critic pairs are trained, and each critic is regressed
# toward a target produced by the OTHER pair:
#     y1 = r + gamma * Q2_target(s', pi1(s'))
#     y2 = r + gamma * Q1_target(s', pi2(s'))
# Each actor maximises its own critic. There is no clipped minimum. Target
# policy smoothing and delayed updates match TD3, so the only difference versus
# CDQ/TD3 is the cross-evaluated double-Q target -- isolating the estimator's
# effect on overestimation, as in the paper's bias experiment.


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)

		self.max_action = max_action


	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(torch.cat([state, action], 1)))
		q = F.relu(self.l2(q))
		return self.l3(q)


class DQ_AC(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2
	):

		# Pair 1
		self.actor1 = Actor(state_dim, action_dim, max_action).to(device)
		self.actor1_target = copy.deepcopy(self.actor1)
		self.actor1_optimizer = torch.optim.Adam(self.actor1.parameters(), lr=3e-4)
		self.critic1 = Critic(state_dim, action_dim).to(device)
		self.critic1_target = copy.deepcopy(self.critic1)
		self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=3e-4)

		# Pair 2
		self.actor2 = Actor(state_dim, action_dim, max_action).to(device)
		self.actor2_target = copy.deepcopy(self.actor2)
		self.actor2_optimizer = torch.optim.Adam(self.actor2.parameters(), lr=3e-4)
		self.critic2 = Critic(state_dim, action_dim).to(device)
		self.critic2_target = copy.deepcopy(self.critic2)
		self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=3e-4)

		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.total_it = 0


	def select_action(self, state):
		# The deployed policy is pair 1's actor.
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor1(state).cpu().data.numpy().flatten()


	def current_value(self, state):
		# Value the deployed policy assigns to its own action, averaged over the
		# two symmetric critics: 0.5 * (Q1 + Q2) at (s, pi1(s)).
		action = self.actor1(state)
		return 0.5 * (self.critic1(state, action) + self.critic2(state, action))


	def _smoothed_next_action(self, actor, next_state, action):
		noise = (
			torch.randn_like(action) * self.policy_noise
		).clamp(-self.noise_clip, self.noise_clip)
		return (actor(next_state) + noise).clamp(-self.max_action, self.max_action)


	def train(self, replay_buffer, batch_size=256):
		self.total_it += 1

		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Cross-evaluated Double Q-learning targets.
			next_action1 = self._smoothed_next_action(self.actor1, next_state, action)
			next_action2 = self._smoothed_next_action(self.actor2, next_state, action)
			target_Q1 = reward + not_done * self.discount * self.critic2_target(next_state, next_action1)
			target_Q2 = reward + not_done * self.discount * self.critic1_target(next_state, next_action2)

		# Critic 1 regresses toward the pair-2 target, critic 2 toward pair-1.
		current_Q1 = self.critic1(state, action)
		critic1_loss = F.mse_loss(current_Q1, target_Q1)
		self.critic1_optimizer.zero_grad()
		critic1_loss.backward()
		self.critic1_optimizer.step()

		current_Q2 = self.critic2(state, action)
		critic2_loss = F.mse_loss(current_Q2, target_Q2)
		self.critic2_optimizer.zero_grad()
		critic2_loss.backward()
		self.critic2_optimizer.step()

		# Delayed policy updates: each actor maximises its own critic.
		if self.total_it % self.policy_freq == 0:

			actor1_loss = -self.critic1(state, self.actor1(state)).mean()
			self.actor1_optimizer.zero_grad()
			actor1_loss.backward()
			self.actor1_optimizer.step()

			actor2_loss = -self.critic2(state, self.actor2(state)).mean()
			self.actor2_optimizer.zero_grad()
			actor2_loss.backward()
			self.actor2_optimizer.step()

			# Update the frozen target models
			pairs = (
				(self.critic1, self.critic1_target),
				(self.critic2, self.critic2_target),
				(self.actor1, self.actor1_target),
				(self.actor2, self.actor2_target),
			)
			for source, target in pairs:
				for param, target_param in zip(source.parameters(), target.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		for name, module in (
			("critic1", self.critic1), ("critic2", self.critic2),
			("actor1", self.actor1), ("actor2", self.actor2),
		):
			torch.save(module.state_dict(), f"{filename}_{name}")


	def load(self, filename):
		self.critic1.load_state_dict(torch.load(f"{filename}_critic1"))
		self.critic2.load_state_dict(torch.load(f"{filename}_critic2"))
		self.critic1_target = copy.deepcopy(self.critic1)
		self.critic2_target = copy.deepcopy(self.critic2)

		self.actor1.load_state_dict(torch.load(f"{filename}_actor1"))
		self.actor2.load_state_dict(torch.load(f"{filename}_actor2"))
		self.actor1_target = copy.deepcopy(self.actor1)
		self.actor2_target = copy.deepcopy(self.actor2)
