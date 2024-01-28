from stochastic_offline_envs.envs.offline_envs.base import BaseOfflineEnv, default_path
from stochastic_offline_envs.envs.connect_four.connect_four_env import ConnectFourEnv
from stochastic_offline_envs.policies.random import RandomPolicy
from stochastic_offline_envs.policies.mixture_policy import EpisodicMixturePolicy, StateMixturePolicy
from stochastic_offline_envs.policies.c4_optimal import C4Optimal
from stochastic_offline_envs.policies.c4_exploitable import C4Specialized, C4MarkovExploitable
from gym import spaces
import re
from pathlib import Path


class ConnectFourOfflineEnv(BaseOfflineEnv):

    def __init__(self, path=default_path('c4data_mdp_random.ds'), horizon=50,
                 n_interactions=int(1e6),
                 exec_dir=default_path('../connect4'), 
                 test_regen_prob=0.8,
                 eps=0.01,
                 data_name=None, 
                 test_only=False,
                 data_dir="offline_data",
                 new_reward=False):
        self.optimal_policy = C4Optimal(exec_dir=exec_dir)
        self.exec_dir = exec_dir
        self.new_reward = new_reward

        if data_name:
            path = default_path(f'{data_name}.ds', data_dir) # c4data_mdp_random, c4data_mdp_random_random, c4data_mdp_20
        if test_regen_prob > 0:
            self.test_opp_policy = self._eps_greedy_policy(eps=test_regen_prob, exec_dir=exec_dir)
        else:
            self.test_opp_policy = self.optimal_policy

        if data_name:
            if "random" not in data_name: # e.g. "c4data_mdp_90", "c4data_mdp_17_mdp_17"
                opts = re.findall(r'\d+', data_name)
                assert len(data_name) > 13
                eps, regen_prob = eval(opts[1]) / 100, eval(opts[2]) / 100
                # # Get eps for decision maker
                # start_ind = data_name.find('_') + 5
                # end_ind = data_name[start_ind:].find('_')
                # eps = eval(data_name[start_ind: start_ind + end_ind]) / 100
                # regen_prob = eval(data_name[data_name.rfind('_') + 1:]) / 100
                # New version: e-greedy adversary
                opp_policy = self._eps_greedy_policy(eps=regen_prob, exec_dir=exec_dir)
            elif data_name == "c4data_mdp_random":
                opp_policy = RandomPolicy(action_space = spaces.Discrete(7))
            elif data_name == "c4data_mdp_random_random":
                eps = 1
                opp_policy = RandomPolicy(action_space = spaces.Discrete(7))
        else:
            raise Exception("Lack data name")

        print("Opt of learner and adv, and test adv:", 1-eps, 1-regen_prob, 1-test_regen_prob)
        print(path)

        env_cls = lambda: ConnectFourEnv(opp_policy, self.optimal_policy, new_reward=new_reward)
        self.test_env_cls = lambda: ConnectFourEnv(self.test_opp_policy, self.optimal_policy, new_reward=new_reward)

        def data_policy_fn():
            # raise AttributeError(
            #     'Environment is attempting to regenerate data, which is not supported in this release. Double check the dataset path.')
            data_specialized_policy = C4Specialized()
            data_eps_greedy = self._eps_greedy_policy(
                eps=eps, exec_dir=exec_dir)
            data_policy = EpisodicMixturePolicy(policies=[data_specialized_policy, data_eps_greedy],
                                                ps=[0.5, 0.5])
            return data_policy

        super().__init__(path, env_cls, data_policy_fn, horizon, n_interactions, test_only)

    def create_test_env(self):
        return ConnectFourEnv(self.test_opp_policy, self.optimal_policy, new_reward=self.new_reward)

    def _eps_greedy_policy(self, eps, exec_dir):
        action_space = spaces.Discrete(7)
        random_policy = RandomPolicy(action_space)

        eps_greedy_policy = StateMixturePolicy(policies=[self.optimal_policy, random_policy],
                                               ps=[1 - eps, eps])

        return eps_greedy_policy
