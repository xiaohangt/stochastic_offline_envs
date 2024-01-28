from stochastic_offline_envs.envs.offline_envs.base import BaseOfflineEnv, default_path
from stochastic_offline_envs.envs.gambling.mstoy import MSToyEnv, NewToyEnv
from stochastic_offline_envs.policies.random import RandomPolicy


class MSToyOfflineEnv(BaseOfflineEnv):

    def __init__(self, path=default_path('mstoy.ds'), horizon=8,
                 n_interactions=int(1e5), new_env=False):
        if new_env:
            env_cls = self.test_env_cls = lambda: NewToyEnv()
            path=default_path('newtoy.ds')
        else:
            env_cls = self.test_env_cls = lambda: MSToyEnv()

        def data_policy_fn():
            test_env = env_cls()
            test_env.action_space
            data_policy = RandomPolicy(test_env.action_space)
            return data_policy

        super().__init__(path, env_cls, data_policy_fn, horizon, n_interactions)
