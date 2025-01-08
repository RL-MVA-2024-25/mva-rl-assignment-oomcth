from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from typing import Protocol


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


envfix = TimeLimit(
    env=FastHIVPatient(
        domain_randomization=False,
        logscale=False
        ), max_episode_steps=200
)

envchange = TimeLimit(
    env=FastHIVPatient(
        domain_randomization=True,
        logscale=False
        ), max_episode_steps=200
)

env = envfix


class ProjectAgent(Protocol):
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cpu")  # Force CPU as per requirements
        self.network = QNetwork(state_dim, action_dim).to(self.device)

    def act(self, observation: np.ndarray) -> int:
        with torch.no_grad():
            # Ensure observation is properly shaped and convert to tensor
            if observation.ndim == 1:
                observation = observation[np.newaxis, :]

            observation_tensor = torch.FloatTensor(observation).to(self.device)
            q_values = self.network(observation_tensor)
            action = q_values.argmax(dim=1).item()
            return action

    def save(self, path: str) -> None:
        # Save only the network state dict to ensure compatibility
        torch.save({
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'network_state': self.network.state_dict()
        }, path)

    def load(self) -> None:
        # Hardcoded path as per requirements
        path = "model.pth"

        # Load the state dict and ensure it's on CPU
        checkpoint = torch.load(path, map_location=self.device)

        # Verify state and action dimensions match
        # assert self.state_dim == checkpoint['state_dim'], "State dimension mismatch"
        # assert self.action_dim == checkpoint['action_dim'], "Action dimension mismatch"

        # Load the network state
        self.network.load_state_dict(checkpoint)
        self.network.eval()  # Set to evaluation mode


class RewardNormalizer:
    def __init__(self, method='standard', gamma=0.99):
        self.method = method
        self.gamma = gamma
        self.running_mean = 0
        self.running_std = 1
        self.running_max = 1e-8
        self.epsilon = 1e-8

    def __call__(self, reward):
        if self.method == 'standard':
            # Update running statistics
            self.running_mean = 0.99 * self.running_mean + 0.01 * reward
            self.running_std = 0.99 * self.running_std + 0.01 * abs(reward - self.running_mean)

            # Normalize
            return (reward - self.running_mean) / (self.running_std + self.epsilon)

        elif self.method == 'log':
            # Log normalization while preserving sign
            sign = np.sign(reward)
            return sign * np.log(1 + abs(reward))

        elif self.method == 'scale':
            # Update maximum seen reward
            # self.running_max = max(self.running_max, abs(reward))

            # Scale between -1 and 1
            return reward / self.running_max

        else:
            return reward / 1e9


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.device = get_device()

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # Optimized tensor creation: convert to numpy arrays first
        states = np.array([s[0] for s in batch])
        actions = np.array([s[1] for s in batch])
        rewards = np.array([s[2] for s in batch])
        next_states = np.array([s[3] for s in batch])
        dones = np.array([s[4] for s in batch])

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.buffer)


# two 512 mid layer = 0.37e10
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.network(x)


def train_agent(env, num_iterations=1_000, buffer_size=10_000,
                batch_size=64, gamma=0.99, epsilon=0.1,
                debug=False, print_freq=100):
    device = get_device()
    print(f"Using device: {device}")
    reward_normalizer = RewardNormalizer(method="none", gamma=gamma)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    best = None

    # Initialize networks and optimizer
    q_network = QNetwork(state_dim, action_dim).to(device)
    target_network = QNetwork(state_dim, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())

    ema_tau = 0.99
    lr = 2e-3
    wd = 0
    optimizer = optim.AdamW(
        q_network.parameters(),
        lr=lr,
        weight_decay=wd
    )

    # Initialize replay buffer with random policy
    replay_buffer = ReplayBuffer(buffer_size)

    # Debug tracking
    episode_rewards = []
    running_reward = 0
    best_reward = float('-inf')

    # Pre-fill buffer with random policy
    print("Pre-filling replay buffer...")
    while len(replay_buffer) < buffer_size:
        state = env.reset()[0]
        done = False
        i = 0
        while i < 200:
            i += 1
            action = env.action_space.sample()
            next_state, reward, done, truncated, info = env.step(action)
            reward = reward_normalizer(reward)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state

    print("Starting training...")
    # Main training loop
    for iteration in range(num_iterations):
        # Bellman update
        if iteration % 4 == 0:
            env = envfix
        else:
            env = envchange

        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

            # Convert numpy arrays to tensors
            state_batch = torch.FloatTensor(states).to(device)
            action_batch = torch.LongTensor(actions).to(device)
            reward_batch = torch.FloatTensor(rewards).to(device)
            next_state_batch = torch.FloatTensor(next_states).to(device)
            done_batch = torch.FloatTensor(dones).to(device)

            # Compute Q values
            current_q_values = q_network(state_batch).gather(1, action_batch.unsqueeze(1))
            next_q_values = target_network(next_state_batch).max(1)[0].detach()
            target_q_values = reward_batch + gamma * next_q_values * (1 - done_batch)

            # Update network
            loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Run episode with e-greedy policy
        state = env.reset()[0]
        done = False
        episode_reward = 0
        i = 0
        while i < 200:
            i += 1
            # e-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    action = q_network(state_tensor).max(1)[1].item()

            next_state, reward, done, truncated, info = env.step(action)
            reward = reward_normalizer(reward)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

        # Debug tracking
        episode_rewards.append(episode_reward)
        running_reward = 0.05 * episode_reward + 0.95 * running_reward
        import copy
        if episode_reward > best_reward:
            best = copy.deepcopy(q_network)
        best_reward = max(best_reward, episode_reward)
        epsilon = max(0.05, 0.1 * (0.9995 ** iteration))

        # Debug printing
        if debug and (iteration + 1) % print_freq == 0:
            print(f"Episode {iteration + 1}")
            print(f"Running reward: {running_reward:.2f}")
            print(f"Episode reward: {episode_reward:.2f}")
            print(f"current eps: {epsilon:.2f}")
            print(f"Best reward so far: {best_reward/1e1:.5f}")
            print("-" * 50)

        # Update target network with EMA

        def update_target_network(target_network, q_network, ema_tau):
            for target_param, q_param in zip(target_network.parameters(),
                                             q_network.parameters()):
                target_param.data.copy_(ema_tau * target_param.data + (1.0 - ema_tau) * q_param.data)

        update_target_network(target_network, q_network, ema_tau)
        if iteration % 100 == 0:
            target_network.load_state_dict(q_network.state_dict())

    if debug:
        # Plot rewards at the end of training
        # plt.figure(figsize=(10, 5))
        # plt.plot(episode_rewards)
        # plt.title('Episode Rewards over Time')
        # plt.xlabel('Episode')
        # plt.ylabel('Reward')
        # plt.show()

        print("\nTraining finished!")
        print(f"Final running reward: {running_reward:.2f}")
        print(f"Best reward achieved: {best_reward:.2f}")

    return best.to('cpu'), episode_rewards


if __name__ == "__main__":
    q_network, rewards = train_agent(env, debug=True, print_freq=100)
    # while max(rewards) < 50:
    #     print("test")
    #     q_network, rewards = train_agent(env, debug=True, print_freq=100)

    torch.save(q_network.state_dict(), 'model.pth')
    torch.save(rewards, "rewards.pth")
    # print(rewards)
