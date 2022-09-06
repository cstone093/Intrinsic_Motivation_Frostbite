from project.agents.random_agent import Random_Agent
from project.hyperparameters.random_hyp import random

agent = Random_Agent(hyp=random)
agent.train()