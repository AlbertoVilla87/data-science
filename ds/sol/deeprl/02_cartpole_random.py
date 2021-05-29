# -*- coding: utf-8 -*-
"""
The random CartPole agent
This environment is from the classic control group and its gist
is to control the platform with a stick attached by its bottom part

In the loop, we sampled a random action, then asked the environmnet to
execute it and return to us the next observation (obs), the reward, and the
done flat. If the episode is over, we stop the loop and show how many
steps we have taken and how much reward has been accumulated
"""

import gym

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    total_reward = 0.0
    total_steps = 0
    obs = env.reset()

while True:
    action = env.action_space.sample()
    obs, reward, done, _ = env.step(action)
    total_reward += reward
    total_steps += 1
    if done:
        break

print("Episode done in %d steps, total reward %.2f" % (total_steps, total_reward))


