from __future__ import print_function

import gym
import numpy as np

import utils

class ModularEnvWrapper(gym.Wrapper):
    """Force env to return fixed shape obs when called .reset() and .step() and removes action's padding before execution"""

    """Also match the order of the actions returned by modular policy to the order of the environment actions"""

    def __init__(self, env, obs_max_len=None, unimal=False):
        super(ModularEnvWrapper, self).__init__(env)
        # if no max length specified for obs, use the current env's obs size
        self.unimal = unimal
        if obs_max_len:
            self.obs_max_len = obs_max_len
        else:
            self.obs_max_len = self.env.observation_space.shape[0]
        self.action_len = self.env.action_space.shape[0]
        if unimal:
            self.num_agents = self.env.num_agents
        else:
            self.num_agents = len(self.env.model.body_names[1:])
        self.agent_obs_size = self.env.observation_space.shape[0] // self.num_agents
        self.max_action = float(self.env.action_space.high[0])
        self.xml = self.env.xml

        # match the order of modular policy actions to the order of environment actions
        # self.motors = utils.getMotorJoints(self.env.xml)
        # self.joints = utils.getGraphJoints(self.env.xml)
        # self.action_order = [-1] * self.num_agents
        # for i in range(len(self.joints)):
        #     assert (
        #         sum([j in self.motors for j in self.joints[i][1:]]) <= 1
        #     ), "Modular policy does not support two motors per body"
        #     for j in self.joints[i]:
        #         if j in self.motors:
        #             self.action_order[i] = self.motors.index(j)
        #             break

    def step(self, action):
        # clip the 0-padding before processing
        # action = action[: self.num_agents]
        # match the order of the environment actions
        # env_action = [None for i in range(len(self.motors))]
        # for i in range(len(action)):
        #     env_action[self.action_order[i]] = action[i]
        if self.unimal:
            env_action = action[:self.num_agents]
        else:
            env_action = action[1:self.num_agents]

        obs, reward, done, info = self.env.step(env_action)
        assert (
            len(obs) <= self.obs_max_len
        ), "env's obs has length {}, which exceeds initiated obs_max_len {}".format(
            len(obs), self.obs_max_len
        )
        obs = np.append(obs, np.zeros((self.obs_max_len - len(obs))))
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        assert (len(obs) <= self.obs_max_len), "env's obs has length {}, which exceeds initiated obs_max_len {}".format(len(obs), self.obs_max_len)
        obs = np.append(obs, np.zeros((self.obs_max_len - len(obs))))
        return obs


class IdentityWrapper(gym.Wrapper):
    """wrapper with useful attributes and helper functions"""

    def __init__(self, env, unimal=False):
        super(IdentityWrapper, self).__init__(env)
        if unimal:
            self.num_agents = self.env.num_agents
        else:
            self.num_agents = len(self.env.model.body_names[1:])
        self.agent_obs_size = self.env.observation_space.shape[0] // self.num_agents
        self.max_action = float(self.env.action_space.high[0])


class ResetWrapper(gym.Wrapper):
    """wrapper to reset env given qpos and qvel"""

    def __init__(self, env):
        super(ResetWrapper, self).__init__(env)

    def frame(self, qpos, qvel):
        self.set_state(qpos, qvel)
