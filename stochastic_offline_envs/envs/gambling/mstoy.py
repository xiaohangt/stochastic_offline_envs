import gym
import numpy as np
from gym import spaces


class MSToyEnv(gym.Env):

    def __init__(self):
        self.action_space = spaces.Discrete(3)
        self.adv_action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(11,), dtype=np.uint8)
        self.state = 0
        self.reward_list = [0, 5, 5, 6, 1, 2, 10, 4]

    def get_obs(self):
        obs = np.zeros(self.observation_space.shape[0])
        obs[self.state] = 1
        return obs

    def reset(self):
        self.state = 0
        return self.get_obs(), None

    def step(self, action):
        reward = 0
        done = False
        adv_action = np.random.choice(2, 1)[0]

        if self.state == 0:
            if adv_action == 0:  
                self.state = 1
            elif adv_action == 1:
                self.state = 2
            self.state += np.clip(action, 0, 1) * 2

            if self.state in [3, 4]:
                reward = self.reward_list[-2 + adv_action]
                done = True
        
        elif self.state in [1, 2]: # 1-> 5, 6, 7; 2-> 8, 9, 10
            self.state = (3 * self.state + 2 + action)
            reward = self.reward_list[self.state - 5]
            done = True

        return self.get_obs(), reward, done, False, {"adv": adv_action}
