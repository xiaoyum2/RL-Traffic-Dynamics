from gym_traffic.envs.traffic_basic_env import TrafficEnv
from gym_traffic.envs.traffic_middle_env import TrafficMidEnv
import numpy as np
import gym
from models.policy import random_policy
from options import parse_options
import logging as log

from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.env_checker import check_env


# Set logger display format
log.basicConfig(format='[%(asctime)s] [INFO] %(message)s', 
                datefmt='%d/%m %H:%M:%S',
                level=log.INFO)

if __name__ == "__main__":
    """Main program."""

    # args, args_str = parse_options()
    # env = gym.make(args.environment_name)
    # env = TrafficEnv()
    env = TrafficMidEnv()
    obs = env.reset()

    # check_env(env)
    # print("Checked!")
   
    # action = random_policy(obs['observation'], obs['desired_goal'],env)
    # obs, reward, done, info = env.step(action)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.1)
    # model = DDPG("MlpPolicy", env, verbose=1, learning_rate=1)

    action, _states = model.predict(obs, deterministic=True)

    # obs, reward, done, info = env.step(action)
    # # env.render()
    # print(env.state)
    # if done:
    #     obs = env.reset()

    model.learn(total_timesteps=1000)
    print("Trained!")
    
    obs = env.reset()
    for i in range(2):
        # print("Road status before:", obs)

        # if i%10==0:
        #     print("Road status before:", obs)
        #     print("Action:", action)
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # env.render()
        print("Action:", action)
        # print("Road status new:", obs)
        print("Reward:", reward)
        print("\n")
        # if i%10==0:
        #     print("Road status new:", obs)
        #     print("Reward:", reward)
        if done:
            obs = env.reset()
    env.close()


    # If we want, we can substitute a goal here and re-compute
    # the reward. For instance, we can just pretend that the desired
    # goal was what we achieved all along.
    # substitute_goal = obs['achieved_goal'].copy()
    # substitute_reward = env.compute_reward( obs['achieved_goal'], substitute_goal, info)
    # print('reward is {}, substitute_reward is {}'.format(reward, substitute_reward))