from stochastic_offline_envs.envs.offline_envs.base import BaseOfflineEnv, default_path
from stochastic_offline_envs.envs.gambling.toy import ToyEnv
from stochastic_offline_envs.policies.random import RandomPolicy


class ToyOfflineEnv(BaseOfflineEnv):

    def __init__(self, path=default_path('toy.ds'), horizon=5, n_interactions=int(1e5)):
        self.env_cls = lambda: ToyEnv()
        self.test_env_cls = lambda: ToyEnv()

        def data_policy_fn():
            test_env = self.env_cls()
            test_env.action_space
            data_policy = RandomPolicy(test_env.action_space)
            return data_policy

        super().__init__(path, self.env_cls, data_policy_fn, horizon, n_interactions)
