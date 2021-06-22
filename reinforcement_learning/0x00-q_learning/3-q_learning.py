#!/usr/bin/env python3
""" Q-learning """
import gym
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """function that performs Q-learning"""
    # List of rewards
    rewards = []

    # 2 For life or until learning is stopped
    for episode in range(episodes):
        # Reset the environment
        state = env.reset()
        done = False
        total_rewards = 0
        step = 0

        while step < max_steps:
            # 3. Choose an action
            action = epsilon_greedy(Q, state, epsilon)

            # Take the action (a) and observe the outcome state(s') and reward
            # (r)
            new_state, reward, done, info = env.step(action)
            # falls in a hole, the reward should be updated to be -1
            if done is True and reward == 0:
                reward = -1
            # Update Q(s,a) += lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            # qtable[new_state,:] : all the actions we can take from new state
            Q[state, action] += alpha * \
                (reward + gamma * np.max(Q[new_state, :]) - Q[state, action])

            total_rewards += reward

            # Our new state is state
            state = new_state

        # If done (if we're dead) : finish episode
            if done:
                break
            step += 1
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (epsilon - min_epsilon) * \
            np.exp(-epsilon_decay * episode)
        rewards.append(total_rewards)
    return Q, rewards
