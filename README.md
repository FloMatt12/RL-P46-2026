# Addressing Function Approximation Error in Actor-Critic Methods

### Setup
Install python \
Create an environmnets and activate it
```
py -m venv .venv
.venv\Scripts\activate
```
\
Install all dependencies (gymnasium[mujoco], numpy, torch)
```
pip install gymnasium[mujoco] numpy torch
```
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

Algorithms which TD3 compares against (PPO, TRPO, ACKTR, DDPG) can be found at [OpenAI baselines repository](https://github.com/openai/baselines). 

### Results
Code is no longer exactly representative of the code used in the paper. Minor adjustments to hyperparamters, etc, to improve performance. Learning curves are still the original results found in the paper.

Learning curves found in the paper are found under /learning_curves. Each learning curve are formatted as NumPy arrays of 201 evaluations (201,), where each evaluation corresponds to the average total reward from running the policy for 10 episodes with no exploration. The first evaluation is the randomly initialized policy network (unused in the paper). Evaluations are peformed every 5000 time steps, over a total of 1 million time steps. 

Numerical results can be found in the paper, or from the learning curves. Video of the learned agent can be found [here](https://youtu.be/x33Vw-6vzso). 

