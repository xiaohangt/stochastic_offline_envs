from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import sys
from tqdm import tqdm
import gymnasium as gym
from stochastic_offline_envs.envs.offline_envs.connect_four_offline_env import ConnectFourOfflineEnv
from stochastic_offline_envs.envs.connect_four.connect_four_env import NewConnectFourEnv
sys.path.insert(1, '/home/xiaohang/castle.home/action-robust-decision-transformer/codebase/esper/stochastic_offline_envs/stochastic_offline_envs/envs/connect_four')

adv_model_path = 'ARDT-Project/ardt-simplest-dataset_combo_train_halfcheetah_v2-2407_0109'
task = ConnectFourOfflineEnv(data_name="c4data_mdp_50_mdp_50")
env = task.test_env_cls()


myEnv_id = "CFourEnv-v0" # It is best practice to have a space name and version number.

gym.envs.registration.register(
    id=myEnv_id,
    entry_point=NewConnectFourEnv,
    max_episode_steps=10, # Customize to your needs.
    reward_threshold=2000 # Customize to your needs.
)


vec_env = make_vec_env(myEnv_id, n_envs=1, seed=0, 
                       env_kwargs={"opponent_policy": task._eps_greedy_policy(eps=0.0, exec_dir=task.exec_dir), 
                                   "optimal_policy": task._eps_greedy_policy(eps=0.0, exec_dir=task.exec_dir)})
model = PPO("MlpPolicy", vec_env, verbose=1)


for _ in tqdm(range(1000)):
    model.learn(total_timesteps=1)
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)


model.save("ppo")
