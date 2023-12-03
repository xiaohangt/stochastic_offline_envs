import pickle
import numpy as np
from copy import deepcopy
from stochastic_offline_envs.envs.offline_envs.connect_four_offline_env import ConnectFourOfflineEnv
from stochastic_offline_envs.envs.connect_four.connect_four_env import GridWrapper
from stochastic_offline_envs.samplers.trajectory_sampler import Trajectory


def get_optimal_traj(env, full=False):
    optimal_action_history = []

    obs_ = []
    actions_ = []
    rewards_ = []

    state = env.reset()
    for t in range(100):
        obs_.append(deepcopy(state))

        action = env.optimal_step(state)
        actions_.append(action)

        state, reward, done, adv_info = env.step(action)
        optimal_action_history.append(action)
        rewards_.append(reward)

        if full and adv_info:
           optimal_traj.append(adv_info['adv_action'])


        if done:
            optimal_trajectory = Trajectory(obs=obs_, actions=actions_, rewards=rewards_, infos=None, policy_infos=None)
            break

    return optimal_action_history, optimal_trajectory


def inject_optimal_traj(env, trajs, opt_ratio=0.5):
    _, optimal_traj = get_optimal_traj(env)
    num_injection = int(len(trajs) * opt_ratio)

    for i, traj in enumerate(trajs):
        if num_injection <= 0:
            break

        if not (traj.actions == optimal_traj.actions and traj.rewards == optimal_traj.actions):
            trajs[i] = optimal_traj
            num_injection -= 1
    return trajs, optimal_traj


def check_num_optimal(trajs, optimal_trajectory):
    count = 0
    for traj in trajs:
        if len(traj.obs) != len(optimal_trajectory.obs):
            continue
        for i, ob in enumerate(traj.obs):
           if not np.all(ob['grid'] == optimal_trajectory.obs[i]['grid']):
               continue
        if traj.actions != optimal_trajectory.actions:
            continue

        count += 1
    return count


if __name__ == "__main__":
    optimal_traj = None
    result_dict = {}
    task = ConnectFourOfflineEnv(test_regen_prob=0.0, data_name="c4data_mdp_50_mdp_50")
    env = task.test_env_cls()
    revised_trajs, optimal_traj = inject_optimal_traj(env, task.trajs)
    print(check_num_optimal(revised_trajs, optimal_traj))

    for learner_prob in ["50"]: #, "45", "40", "35", "30", "25", "20", "10", "0"]:
        for adv_prob in ["50", "45", "40", "35", "30", "25", "20", "10", "0"]:
            count = 0
            d_name = f"c4data_mdp_{learner_prob}_mdp_{adv_prob}"
            task = ConnectFourOfflineEnv(test_regen_prob=0.0, data_name=d_name)
            env = task.test_env_cls()

            if not optimal_traj:
                optimal_traj, optimal_trajectory = get_optimal_traj(env)

            result_dict[f"{learner_prob}_{adv_prob}"] = check_num_optimal(trajs, optimal_trajectory)

    pickle.dump(result_dict, open(f'results/data_profile.pkl', 'wb'))
