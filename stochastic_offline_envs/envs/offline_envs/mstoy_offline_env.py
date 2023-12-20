from stochastic_offline_envs.envs.offline_envs.base import BaseOfflineEnv, default_path
from stochastic_offline_envs.envs.gambling.mstoy import MSToyEnv
from stochastic_offline_envs.envs.gambling.new_mstoy import NewMSToyEnv
from stochastic_offline_envs.policies.random import RandomPolicy


class MSToyOfflineEnv(BaseOfflineEnv):

    def __init__(self, horizon=8,
                 n_interactions=int(1e5), new=False, data_name='mstoy.ds'):
        self.test_env_cls = env_cls = MSToyEnv if not new else NewMSToyEnv
        
        path = default_path(f'{data_name}.ds')

        def data_policy_fn():
            test_env = env_cls()
            test_env.action_space
            data_policy = RandomPolicy(test_env.action_space)
            return data_policy

        super().__init__(path, env_cls, data_policy_fn, horizon, n_interactions)
