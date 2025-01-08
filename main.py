import random
import os
import numpy as np
import torch
import train
from evaluate import evaluate_HIV, evaluate_HIV_population
from train import ProjectAgent  # Replace DummyAgent with your agent implementation
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(seed=42)
    q_network, rewards = train.train_agent(
        TimeLimit(
                    env=HIVPatient(
                        domain_randomization=False,
                        logscale=False
                        ), max_episode_steps=200
                ), debug=True, print_freq=100)
    # while max(rewards) < 50:
    #     print("test")
    #     q_network, rewards = train_agent(env, debug=True, print_freq=100)

    torch.save(q_network.state_dict(), 'model.pth')
    torch.save(rewards, "rewards.pth")
    # print(rewards)

    # Initialization of the agent. Replace DummyAgent with your custom agent implementation.
    state_dim = 6
    action_dim = 4
    agent = ProjectAgent(state_dim, action_dim)
    agent.load()
    # Keep the following lines to evaluate your agent unchanged.
    score_agent: float = evaluate_HIV(agent=agent, nb_episode=1)
    score_agent_dr: float = evaluate_HIV_population(agent=agent, nb_episode=15)
    with open(file="score.txt", mode="w") as f:
        f.write(f"{score_agent}\n{score_agent_dr}")
