import gym
import numpy as np
import matplotlib.pyplot as plt
import  collections
 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

class NN(Model):
    def __init__(self, state_shape, num_actions, lr):
        super(NN, self).__init__()
        self.state_shape = state_shape
        self.num_actions = num_actions
        self.lr = lr

        state = Input(shape=(state_shape,))
        x = Dense(24)(state)
        x = Activation("relu")(x)
        x = Dense(self.num_actions)(x)
        out = Activation("softmax")(x)
        self.model = Model(inputs=state, outputs=out)
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.lr))

    def train(self, states, q_values):
        self.model.fit(states, q_values, verbose=0)

    def predict(self, state):
        return self.model.predict(state)

    def update_model(self, other_model):
        self.model.set_weights(other_model.get_weights())

    def load_model(self, path):
        self.model.load_weights(path)

    def save_model(self, path):
        self.model.save_weights(path) 
   
class NN_A2C(Model):
    def __init__(self, num_observations, num_actions, num_values, lr_actor, lr_critic):
        super(NN_A2C, self).__init__()
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.num_values = num_values
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        state = Input(shape=(num_observations,)) # (batch_size, num_of_observations)
        x = Dense(24)(state)
        x = Activation("relu")(x)

        actor_x = Dense(self.num_actions)(x)
        actor_out = Activation("softmax")(actor_x)
        self.actor = Model(inputs=state, outputs=actor_out)
        self.actor.summary()
        self.actor.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.lr_actor))

        critic_x = Dense(self.num_values)(x)
        critic_out = Activation("linear")(critic_x)
        self.critic = Model(inputs=state, outputs=critic_out)
        self.critic.summary()
        self.critic.compile(loss="mse", optimizer=Adam(lr=self.lr_critic))

    # Actor Functions
    def train_actor(self, states, actions):
        self.actor.fit(states, actions, verbose=0)

    def predict_actor(self, states):
        return self.actor.predict(states)

    # Critic Functions
    def train_critic(self, states, values):
        self.critic.fit(states, values, verbose=0)

    def predict_critic(self, states):
        return self.critic.predict(states)