from stochastic_offline_envs.samplers.trajectory_sampler import TrajectorySampler
from os import path
import pickle
import os


class BaseOfflineEnv:

    def __init__(self, p, env_cls, data_policy, horizon, n_interactions, test=False):
        self.env_cls = env_cls
        self.data_policy = data_policy
        self.horizon = horizon
        self.n_interactions = n_interactions
        self.p = p
        if test:
            return

        if self.p is not None and path.exists(self.p):
            print('Dataset file found. Loading existing trajectories.')
            with open(self.p, 'rb') as file:
                self.trajs = pickle.load(file)
        else:
            print('Dataset file not found. Generating trajectories.')
            self.generate_and_save()

    def generate_and_save(self):
        self.trajs = self.collect_trajectories()

        if self.p is not None:
            os.makedirs(path.dirname(self.p), exist_ok=True)
            with open(self.p, 'wb') as file:
                pickle.dump(self.trajs, file)
                print('Saved trajectories to dataset file.')

    def collect_trajectories(self):
        data_policy = self.data_policy()
        sampler = TrajectorySampler(env_cls=self.env_cls,
                                    policy=data_policy,
                                    horizon=self.horizon)
        trajs = sampler.collect_trajectories(self.n_interactions)
        return trajs


def default_path(name, is_data=True):
    # Get the path of the current file
    file_path = path.dirname(path.realpath(__file__))
    # Go up 3 directories
    root_path = path.abspath(path.join(file_path, '..', '..', '..'))
    if is_data:
        # Go to offline data directory
        full_path = path.join(root_path, 'offline_data')
    else:
        full_path = root_path
    # Append the name of the dataset
    return path.join(full_path, name)
