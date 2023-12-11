import gym
import numpy as np
from gym import spaces


class ToyEnv(gym.Env):

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.adv_action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(7,), dtype=np.uint8
        )
        self.state = 0

    def get_obs(self):
        obs = np.zeros(self.observation_space.shape[0])
        obs[self.state] = 1
        return obs

    def reset(self):
        self.state = 0
        return self.get_obs(), None

    def step(self, action):
        adv_action = 0
        if self.state == 0:
            reward = 0
            done = False
            adv_action = np.random.choice(3, 1)[0]
            if adv_action == 0:  
                self.state = 1
            elif adv_action == 1:
                self.state = 2
            else:
                self.state = 3
            self.state += action * 3
        elif self.state == 1:
            reward = 0
            done = True
        elif self.state == 2:
            reward = 4
            done = True
        elif self.state == 3:
            reward = 6
            done = True
        elif self.state == 4:
            reward = 6
            done = True
        elif self.state == 5:
            reward = 1
            done = True
        elif self.state == 6:
            reward = 2
            done = True
        return self.get_obs(), reward, done, False, {"adv": adv_action}
