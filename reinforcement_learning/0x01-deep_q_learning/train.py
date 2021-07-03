#!/usr/bin/env python3
"""train an agent that can play Atari’s Breakout"""
import gym
import numpy as np
import keras as K


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


# def train():
#     env = gym.make("SpaceInvaders-v0")
#     height, width, channels = env.observation_space.shape
#     action_size = env.action_space.n
#     state_size = env.observation_space.n
#     Q_table = np.zeros((state_size, action_size))



# --------------------------------------------------------------------------- #

WINDOW_WIDTH = 3
MAX_STEPS = 10000000
WARMUP_STEPS = 50000
CALLBACK_INTERVAL = 250000
LOG_INTERVAL = 1000
MAX_EPSILON = 1
MIN_EPSILON = .1
TEST_EPSILON = .2
LEARNING_RATE = 1e-4
ENV_NAME = "Breakout-v0"


# --------------------------------------------------------------------------- #

env = gym.make(ENV_NAME)
nb_actions = env.action_space.n
input_shape = env.observation_space.shape


# --------------------------------------------------------------------------- #
def build_model(input_shape, nb_actions):
    """Build a convolutional network model
    Arguments:
        input_shape {tuple} -- Contains the shape of the inputs
        nb_actions {int} -- The number of available actions
    Returns:
        keras.Model -- keras model instance
    """
    model = Sequential()
    model.add(Input(shape=(WINDOW_WIDTH,) + input_shape))
    model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
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


model = build_model(input_shape, nb_actions)


# --------------------------------------------------------------------------- #
def build_agent(model, nb_actions):
    """Builds an agent
    Returns:
        keras.DQNAgent -- DQNAgent instance
    """
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=MAX_EPSILON,
        value_min=MIN_EPSILON,
        value_test=TEST_EPSILON,
        nb_steps=MAX_STEPS
    )
    memory = SequentialMemory(
        limit=MAX_STEPS,
        window_length=WINDOW_WIDTH
    )
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        enable_dueling_network=True,
        dueling_type='avg',
        nb_actions=nb_actions,
        nb_steps_warmup=WARMUP_STEPS
    )
    dqn.compile(Adam(learning_rate=LEARNING_RATE), metrics=['mae'])
    return dqn


dqn = build_agent(model, nb_actions)


# --------------------------------------------------------------------------- #
def train_agent(dqn, env, weights_filename):
    """Trains an agent
    """
    checkpoint_weights_filename = 'dqn_' + ENV_NAME + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(ENV_NAME)

    callbacks = [
        ModelIntervalCheckpoint(
            checkpoint_weights_filename,
            interval=CALLBACK_INTERVAL
        )
    ]
    callbacks += [FileLogger(log_filename, interval=LOG_INTERVAL)]
    dqn.fit(
        env,
        callbacks=callbacks,
        nb_steps=MAX_STEPS,
        log_interval=LOG_INTERVAL
    )
    dqn.save_weights(
        weights_filename,
        overwrite=True
    )
