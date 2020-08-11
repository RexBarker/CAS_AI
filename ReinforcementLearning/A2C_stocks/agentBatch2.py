import gym
import numpy as np
import matplotlib.pyplot as plt
import collections
import random
        
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

from nn import *

class Agent:
    def __init__(self, env):
        self.env = env
        self.num_observations = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.n
        self.num_values = 1
        self.gamma = 0.99
        self.lr_actor = 1e-3
        self.lr_critic = 5e-3
        self.actions,  self.rewards, self.states, self.next_states = [], [], [], []
        self.model = NN_A2C(self.num_observations, self.num_actions, self.num_values, self.lr_actor, self.lr_critic)

    def get_action(self, state):
        policy = self.model.predict_actor(state)[0]
        action = np.random.choice(self.num_actions, p=policy)
        return action

    def discounted_rewards(self):
        discounted_rewards = np.zeros_like(self.rewards)
        sum_t = 0.0
        for t in reversed(range(len(self.rewards))):
            sum_t = sum_t * self.gamma + self.rewards[t]
            discounted_rewards[t] = sum_t
        return discounted_rewards

    def normalize_discounted(self, discounted_rewards):
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        return discounted_rewards

    def append_sample(self, state, next_state, action, reward):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.rewards.append(reward)

    def update_policy(self):
        states = np.reshape(self.states,(len(self.states),self.num_observations))
        next_states = np.reshape(self.next_states,(len(self.next_states),self.num_observations))
        advantages_a = np.zeros((len(self.states),self.num_actions))
        values_a = np.zeros((len(self.states),self.num_actions))

        n = len(self.states)
        for i in range(len(self.states)):
            state = np.array([states[i]])
            next_state = np.array([next_states[i]])
            reward = self.rewards[i]
            action = self.actions[i]

            values = np.zeros((1, self.num_values))
            advantages = np.zeros((1, self.num_actions))

            value = self.model.predict_critic(state)[0]
            next_value = self.model.predict_critic(next_state)[0]

            if i == len(self.states)-1:
                advantages[0][action] = reward - value
                values[0][0] = reward
            else:
                advantages[0][action] = (reward + self.gamma * next_value) - value
                values[0][0] = reward + self.gamma * next_value

            advantages_a[i,:] = advantages
            values_a[i] = values

            self.model.train_actor(state, advantages)
            self.model.train_critic(state, values)

        #advantages_a = np.reshape(advantages_a,(1,))

    def train(self, num_episodes):
        total_rewards = []
        for episode in range(num_episodes):
            total_reward = 0.0
            state = self.env.reset()
            state = np.reshape(state, (1, self.num_observations))

            while True:
                action = self.get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, (1, self.num_observations))
                reward = reward if not done or total_reward == 499 else -100
                self.append_sample(state, next_state, action, reward)
                total_reward += reward
                state = next_state

                if done:
                    self.update_policy()
                    total_reward = total_reward if total_reward == 500 else total_reward + 100
                    total_rewards.append(total_reward)
                    mean_total_rewards = np.mean(total_rewards[-min(len(total_rewards), 10):])

                    print("Episode: ", episode+1,
                        " Total Reward: ", total_reward,
                        " Mean: ", mean_total_rewards)
                    
                    if mean_total_rewards >= 495.0:
                        return total_rewards
                    break
        return total_rewards


    def play(self, num_episodes, render=True):
        for episode in range(num_episodes):
            state = self.env.reset()
            while True:
                state = np.reshape(state, (1, self.num_observations))
                if render:
                    self.env.render()
                action = self.get_action(state)
                state, reward, done, _ = self.env.step(action)
                if done:
                    break

if __name__ == "__main__":
    env = gym.make("CartPole-v1")
    agent = Agent(env)
    total_rewards = agent.train(num_episodes=500)

    plt.plot(range(len(total_rewards)), total_rewards, color="blue")
    plt.savefig("./agent A2C.png")

    input("Play?")
    agent.play(num_episodes=15, render=True)