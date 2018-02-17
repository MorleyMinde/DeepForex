from keras.layers import Dense,Dropout, Activation,TimeDistributed, Flatten, Convolution1D,Convolution2D,Permute,LSTM,MaxPooling1D,Reshape,BatchNormalization,Input
from keras.optimizers import RMSprop
from keras.models import Sequential,load_model
from keras.optimizers import Adam
import numpy as np
import random
import os
from collections import deque
import json

class Agent():
    def __init__(self, state_size, action_size,dir):
        self.weight_backup = "{}/weights.h5".format(dir)
        self.config = json.load(open('{}/config.json'.format(dir)))
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.learning_rate = 0.001
        self.gamma = 0.9
        self.exploration_rate = 1.0
        self.exploration_min    = 0.01
        self.exploration_decay  = 0.995
        self.brain              = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        # model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(32, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(10, activation='relu'))
        # model.add(Dense(self.action_size, activation='linear'))
        # model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        #
        if self.config['network'] == "CNNRNN":
            model.add(LSTM(10,  return_sequences=True,input_shape=(30,8)))
            model.add(Dropout(0.5))
            model.add(LSTM(50,  return_sequences=True))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(50))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(10, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss=self.config['loss'], optimizer=Adam(lr=self.learning_rate))
        else:
            PERIODS_PER_X = 30
            # model.add(LSTM(32,  return_sequences=True,input_shape=(30,8)))
            # model.add(Dropout(0.5))
            # model.add(LSTM(32))
            # model.add(Dropout(0.5))
            # model.add(Dense(32, activation='relu'))
            # model.add(Dropout(0.5))
            # model.add(Dense(10, activation='relu'))
            # model.add(Dense(self.action_size, activation='linear'))
            # model.compile(loss=self.config['loss'], optimizer=Adam(lr=self.learning_rate))
            # model.add(Reshape((1, PERIODS_PER_X, self.state_size), input_shape=(PERIODS_PER_X, self.state_size)))
            # model.add(Input((PERIODS_PER_X, self.state_size)))
            model.add(Convolution1D(128, 3, padding='same',activation='relu',input_shape=(PERIODS_PER_X,8)))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Convolution1D(64, 3, activation='relu', padding="same"))
            model.add(MaxPooling1D(pool_size=2))
            # model.add(Reshape((PERIODS_PER_X, self.state_size)))
            model.add(LSTM(64, return_sequences=True))
            model.add(Dropout(0.25))
            model.add(LSTM(32, return_sequences=True))
            model.add(Dropout(0.5))
            model.add(Dense(16, activation='relu'))
            model.add(Dense(self.action_size, activation='softmax'))
            # if doesn't fit as well as other one, try adding more layers, or try using the old model with only price information (KISS)

            model.compile(loss=self.config['loss'], optimizer=Adam(lr=self.learning_rate))
        # Create the model based on the information above
        #model.compile(loss='mean_squared_error', optimizer='adam')
        if os.path.isfile(self.weight_backup):
            #load_model(self.weight_backup)
            model.load_weights(self.weight_backup,True)
            self.exploration_rate = self.exploration_min
        return model
    def save_model(self):
        self.brain.save(self.weight_backup)
    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        action = np.argmax(act_values[0])
        if action > 3:
           print(action,act_values,act_values[0])
        return action
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
              target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def predict(self,state):
        return self.brain.predict(state)