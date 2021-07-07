#!/usr/bin/env python3
"""
play Breakout using deep Q-learning
"""

import numpy as np
import gym
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Convolution2D
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

WINDOW_LENGTH = 4
env = gym.make('Breakout-v0')
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
print(nb_actions)
input_shape_ = env.observation_space.shape
env.unwrapped.get_action_meanings()


def build_model(input_shape, nb_actions):
    """build the CNN model to train it"""
    model = Sequential()

    model.add(
        Convolution2D(
            32, (8, 8), strides=(
                4, 4), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    return model


model = build_model(
    input_shape=(
        (WINDOW_LENGTH,
         ) + input_shape_),
    nb_actions=nb_actions)

model.summary()


def build_agent(model, actions):
    """build the agent with specific policy"""
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.,
        value_min=.1,
        value_test=.2,
        nb_steps=10000)
    memory = SequentialMemory(limit=100000, window_length=4)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000
                   )
    return dqn


dqn = build_agent(model, nb_actions)


def play(env):
    """play breakout using trained model save"""
    from colabgymrender.recorder import Recorder
    dqn.load_weights('policy.h5')

    """recorde and visualization the game"""
    directory = './video'
    env = Recorder(env, directory)

    observation = env.reset()
    terminal = False
    while not terminal:
        action = env.action_space.sample()
        observation, reward, terminal, info = env.step(action)
    env.play()


play(env=env)
