# -*- coding: utf-8 -*-
"""
Randomly behaving agent and become more familiar with the basic concepts of RL
Description: we define an environment that will give the agent random rewards
for a limited number of steps, regardless of the agent's actions.
"""

import random
from typing import List

# Environment

class Environment:
    def __init__(self):
        self.steps_left = 10

    def get_observations(self) -> List[float]:
        """
        Return the current environment's observation to the agent.
        :return: environment's observation to the agent
        :rtype: List[float]
        """
        return [0.0, 0.0, 0.0]

    def get_actions(self) -> List[int]:
        """
        Allows the agent to query the set of actions it can execute.
        :return: set of actions
        :rtype: List[int]
        """
        return [0, 1]

    def is_done(self) -> bool:
        """
        Check end of the episode to the agent.
        :return: Check the end
        :rtype: bool
        """
        return self.steps_left == 0

    def action(self, action: int) -> float:
        """
        Handles an agent's action and return the rewards for this action
        :param int action: type of action
        :raises Exception: Game over
        :return: reward
        :rtype: float
        """
        if self.is_done():
            if self.is_done():
                raise Exception("Game is over")
        self.steps_left -= 1
        return random.random()

# Agent

class Agent:
    def __init__(self):
        self.total_reward = 0.0

    def step(self, env:Environment):
        """
        The step function accepts the environment instance as an argument and allows
        the agent to perfoirm the following actions:
            - Observe the environment
            - Make a decision about the action to take based on the observations
            - Submit the action to the environment
            - Get the reward for the current step
        :param Environment env: [description]
        """
        current_obs = env.get_observations()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward

# Main

if __name__ == "__main__":
    env = Environment()
    agent = Agent()
    while not env.is_done():
        agent.step(env)

    print("Total reward got: %.4f" % agent.total_reward)





