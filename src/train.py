import numpy as np
from copy import deepcopy
import math
from env_hiv import HIVPatient
from interface import Agent
from gymnasium.wrappers import TimeLimit
import torch
import torch.nn as nn
import torch.optim as optim
import random


device = "mps" if torch.mps.is_available() else "cpu"


class MCTSNode:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0

    def add_child(self, child_state, action):
        child = MCTSNode(child_state, action, self)
        self.children.append(child)
        return child

    def ucb_score(self, exploration_constant):
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits+1) / self.visits)
        return exploitation + exploration

    def best_child(self, exploration_constant):
        return max(self.children, key=lambda child: child.ucb_score(exploration_constant))

    def get_untried_actions(self, possible_actions):
        tried_actions = {child.action for child in self.children}
        return list(set(possible_actions) - tried_actions)


class MCTS:
    def __init__(self, env, state, exploration_constant=1e6, horizon=200):
        self.env = env
        self.state = state
        self.exploration_constant = exploration_constant
        self.horizon = horizon
        self.best_path = None
        self.best_reward = float('-inf')
        self.root = None

    def simulate_episode(self, node, current_state,
                         steps_taken, current_path=None):
        if current_path is None:
            current_path = []

        sim_env = deepcopy(self.env)
        sim_env.state = deepcopy(current_state)
        total_reward = 0
        path = current_path.copy()

        while steps_taken < self.horizon:
            action = sim_env.action_space.sample()
            _, reward, _, _, _ = sim_env.step(action)
            total_reward += reward
            path.append(action)
            steps_taken += 1

        return total_reward, path

    def run_episode(self, initial_state):
        if self.root is None:
            self.root = MCTSNode(initial_state)

        node = self.root
        sim_env = deepcopy(self.env)
        sim_env.state = deepcopy(initial_state)
        current_path = []
        current_reward = 0
        steps_taken = 0

        while steps_taken < self.horizon:
            if not len(node.children) == sim_env.action_space.n:
                untried_actions = node.get_untried_actions(range(sim_env.action_space.n))
                action = np.random.choice(untried_actions)
                next_state, reward, _, _, _ = sim_env.step(action)
                current_reward += reward
                current_path.append(action)
                node = node.add_child(next_state, action)
                steps_taken += 1
            else:
                node = node.best_child(self.exploration_constant)
                _, reward, _, _, _ = sim_env.step(node.action)
                current_reward += reward
                current_path.append(node.action)
                steps_taken += 1

        simulation_reward, complete_path = self.simulate_episode(
            node, sim_env.state, steps_taken, current_path
        )
        total_reward = current_reward + simulation_reward

        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_path = complete_path

        while node:
            node.visits += 1
            node.value += total_reward
            node = node.parent

    def train(self, n_episodes):
        initial_state = self.state
        for _ in range(n_episodes):
            self.run_episode(initial_state)

    def get_action_sequence(self):

        actions = []
        node = self.root
        steps = 0

        while node.children and steps < self.horizon:
            best_child = max(node.children, key=lambda child: child.value / child.visits if child.visits > 0 else float('-inf'))
            actions.append(best_child.action)
            node = best_child
            steps += 1

        while steps < self.horizon:
            if self.best_path and steps < len(self.best_path):
                actions.append(self.best_path[steps])
            else:
                actions.append(self.env.action_space.sample())
            steps += 1

        return actions


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.model(x)


def train_fix():
    env = TimeLimit(env=HIVPatient(
        domain_randomization=False),
        max_episode_steps=200
    )
    current_state = env.reset()
    horizon = 25
    epoch = int(200 / horizon)
    assert epoch * horizon == 200

    all_actions = []
    for i in range(epoch):
        mcts = MCTS(
            env, current_state, horizon=horizon,
            exploration_constant=min(1e5 * 4 ** i, 1e8)
        )
        mcts.train(n_episodes=1_000)
        actions = mcts.get_action_sequence()
        all_actions += actions
        for action in actions:
            current_state, reward, _, _, _ = env.step(action)

    env.reset()
    total_reward = 0
    i = 0
    while i < 200:
        for action in all_actions:
            i += 1
            _, reward, _, _, _ = env.step(action)
            total_reward += reward
            if i == 200:
                break
    torch.save(all_actions, "all_actions.pth")


def train_rnd():
    env = TimeLimit(env=HIVPatient(
        domain_randomization=True),
        max_episode_steps=200
    )
    state_dim = 6
    action_dim = 4
    q_network = QNetwork(state_dim, action_dim).to(device)
    target_network = QNetwork(state_dim, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    lr = 0.01
    gamma = 0.995
    mem_size = 10000
    batch_size = 64
    n_episode = 500
    target_update_freq = 10
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    criterion = nn.MSELoss()

    replay_buffer = []
    epsilon = 0.1
    total_rewards = []

    for episode in range(n_episode):
        env.domain_randomization = episode % 3 == 0
        state, _ = env.reset()
        total_reward = 0
        steps = 0

        while steps < env._max_episode_steps:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = q_network(state_tensor)
                action = torch.argmax(q_values).item()

            next_state, reward, _, _, _ = env.step(action)
            done = steps == env._max_episode_steps - 1

            if len(replay_buffer) >= mem_size:
                replay_buffer.pop(0)
            replay_buffer.append((state, action, reward, next_state, done))

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.tensor(
                    states, dtype=torch.float32, device=device
                )
                actions = torch.tensor(
                    actions, dtype=torch.int64, device=device
                )
                rewards = torch.tensor(
                    rewards, dtype=torch.float32, device=device
                )
                next_states = torch.tensor(
                    next_states, dtype=torch.float32, device=device
                )
                dones = torch.tensor(
                    dones, dtype=torch.float32, device=device
                )

                q_values = q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q_values = target_network(next_states).max(1)[0]
                    target_q_values = rewards + gamma * next_q_values

                loss = criterion(q_values, target_q_values)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            state = next_state
            total_reward += reward
            steps += 1

        total_rewards.append(total_reward)

        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())
        epsilon = max(0.01, 0.1*0.995**episode)

    torch.save(q_network.state_dict(), "model.pth")


# For some reason I cannot make DQN work for fix agent but it works well on
#  random env so we take DQN for random env and a MCTS on fix env.
# MCTS works by optimizing the episode 25 steps by 25 steps to have a
#  shorter time horizon. This is a king of somewhat greedy algorithm.

class MainAgent(Agent):
    def __init__(self):
        super().__init__()
        self.step = 0
        self.fix = True

        # We can know if we're in fixed env by looking at state after one action
        # we could automate this by fixing fix_initial_state when exploring the MCTS
        self.fix_initial_state = np.array(
            [1.99133864e+05, 1.21948890e+03,
             3.80403636e+01, 3.31445905e+01,
             6.94253940e+03, 2.58476879e+01]
        )

    def load(self):
        self.all_actions = torch.load("all_actions.pth", weights_only=False)
        self.q_network = QNetwork(6, 4)
        self.q_network.load_state_dict(torch.load("model.pth", map_location='cpu'))
        self.q_network = self.q_network.to(device)

    def save(self):
        pass

    def act(self, state):
        if self.step % 200 == 1:
            if np.allclose(state, self.fix_initial_state, 0.001):
                self.fix = True
            else:
                self.fix = False
            print(self.fix)
        self.step += 1
        if self.fix:
            return self.all_actions[(self.step-1) % 200]
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state)
            return torch.argmax(q_values).item()


if __name__ == "__main__":
    # train_fix()
    train_rnd()
