#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author: edwardahn

Environment for training a local planner to move in a straight line.
"""

import numpy as np

from rllab.spaces import Box

from aa_simulation.envs.base_env import VehicleEnv
from aa_simulation.misc.utils import normalize_angle


class StraightEnv(VehicleEnv):
    """
    Simulation environment for an RC car following a straight
    line trajectory. The reward function encourages the agent to
    move right on the line y=0 for all time.
    """

    def __init__(
            self,
            target_velocity=1.0,
            dt=0.035,
            model_type='BrushTireModel',
            robot_type='RCCar'
    ):
        """
        Initialize super class parameters, obstacles and radius.
        """
        super(StraightEnv, self).__init__(
            target_velocity=target_velocity,
            dt=dt,
            model_type=model_type,
            robot_type=robot_type
        )
        self.robot_type = robot_type

        # Reward function parameters
        self._lambda1 = 0.25


    @property
    def observation_space(self):
        """
        Define the shape of input vector to the neural network.
        """
        return Box(low=-np.inf, high=np.inf, shape=(5,))


    @property
    def get_initial_state(self):
        """
        Get initial state of car when simulation is reset.
        """
        # Randomly initialize state for better learning
        if self.robot_type == 'RCCar':
            y = np.random.uniform(-0.25, 0.25)
            yaw = np.random.uniform(-np.pi/3, np.pi/3)
            x_dot = np.random.uniform(0, 1.3)
            y_dot = np.random.uniform(-0.6, 0.6)
            yaw_dot = np.random.uniform(-2.0, 2.0)
        elif self.robot_type == 'MRZR':
            y = np.random.uniform(-0.25, 0.25)
            yaw = np.random.uniform(-np.pi/3, np.pi/3)
            x_dot = np.random.uniform(0, 2.0)
            y_dot = np.random.uniform(-0.6, 0.6)
            yaw_dot = np.random.uniform(-0.3, 0.3)
        else:
            raise ValueError('Unrecognized robot type')

        state = np.zeros(6)
        state[1] = y
        state[2] = yaw
        state[3] = x_dot
        state[4] = y_dot
        state[5] = yaw_dot
        return state


    def get_reward(self, state, action):
        """
        Reward function definition.
        """
        _, y, _, x_dot, y_dot, _ = state
        velocity = np.sqrt(x_dot**2 + y_dot**2)
        distance = y

        reward = -np.absolute(distance)
        reward -= self._lambda1 * (velocity - self.target_velocity)**2

        info = {}
        info['dist'] = distance
        info['vel'] = velocity
        return reward, info


    @staticmethod
    def project_line(state, x0, y0, angle):
        """
        Note that this policy is trained to follow a straight line
        to the right (y = 0). To follow an arbitrary line, use this
        function to transform the current absolute state to a form
        that makes the policy believe the car is moving to the right.

        :param state: Current absolute state of robot
        :param x0: x-value of start of line to follow
        :param y0: y-value of start of line to follow
        :param angle: Angle of line to follow
        """
        x, y, yaw, x_dot, y_dot, yaw_dot = state
        angle = normalize_angle(angle)

        current_angle = np.arctan2((y - y0), (x - x0))
        projected_angle = normalize_angle(current_angle - angle)
        dist = np.sqrt((x - x0)**2 + (y - y0)**2)

        new_x = dist * np.cos(projected_angle)
        new_y = dist * np.sin(projected_angle)
        new_yaw = normalize_angle(yaw - angle)

        return np.array([new_x, new_y, new_yaw, x_dot, y_dot, yaw_dot])


    def state_to_observation(self, state):
        """
        Prepare state to be read as input to neural network.
        """
        _, y, yaw, x_dot, y_dot, yaw_dot = state
        yaw = normalize_angle(yaw)
        return np.array([y, yaw, x_dot, y_dot, yaw_dot])

