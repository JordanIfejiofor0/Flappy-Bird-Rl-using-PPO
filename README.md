# Flappy Bird RL — PPO

A reinforcement learning project where I trained an agent to play a custom Flappy Bird environment using **Proximal Policy Optimization (PPO)** in PyTorch.

## What is the project about?

The project explores how an RL agent can learn to play a simple game through **trial and error**, without being explicitly programmed with the strategy for playing.

## What does it try to solve?

The goal is to train the agent to **navigate through the pipes and survive for as long as possible** by learning which actions to take from its observations of the game.

## Architecture

The agent uses an **Actor-Critic neural network**:

* **Actor** — predicts the probability of each possible action.
* **Critic** — estimates the value of the current state.
* **PPO** — updates the policy while keeping training stable.

The agent interacts with the environment, collects experience, and uses that experience to improve its policy over multiple training iterations.

## Tech Stack

* **Python**
* **PyTorch**
* **Pygame**
* **NumPy**
* **PPO (Proximal Policy Optimization)**

## Future Improvements

* Improve the observation space and reward function.
* Experiment with different network architectures and hyperparameters.
* Improve training efficiency and stability.
* Test the agent in more complex environments.
