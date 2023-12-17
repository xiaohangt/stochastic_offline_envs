import gym
import numpy as np
from gym import spaces


class MSToyEnv(gym.Env):

    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.adv_action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(13,), dtype=np.uint8)
        self.state = 0

    def get_obs(self):
        obs = np.zeros(self.observation_space.shape[0])
        obs[self.state] = 1
        return obs

    def reset(self):
        self.state = 0
        return self.get_obs()

    def step(self, action):
        reward = 0
        done = False
        adv_action = np.random.choice(2, 1)[0]

        if self.state == 0:
            if action == 0:
                if adv_action == 0:
                    self.state = 1
                elif adv_action == 1:
                    self.state = 2
            elif action == 1:
                if adv_action == 0:
                    self.state = 3
                elif adv_action == 1:
                    self.state = 4
                    reward = 15.0
                    done = True
        elif self.state == 1:
            if action == 0:
                if adv_action == 0:
                    self.state = 5
                    reward = -1.0
                    done = True
                elif adv_action == 1:
                    self.state = 6
                    reward = 0.0
                    done = True
            elif action == 1:
                if adv_action == 0:
                    self.state = 7
                    reward = -3.0
                    done = True
                elif adv_action == 1:
                    self.state = 8
                    reward = 4.0
                    done = True
        elif self.state == 2:
            if action == 0:
                self.state = 9
                reward = -2.0
                done = True
            elif action == 1:
                self.state = 10
                reward = 10.0
                done = True
        elif self.state == 3:
            if action == 0:
                self.state = 11
                reward = -5.0
                done = True
            elif action == 1:
                self.state = 12
                reward = -1.5
                done = True
        else:
            raise RuntimeError("Invalid state")
            
        return self.get_obs(), reward, done, {"adv": adv_action}
