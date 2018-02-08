from keras.layers import Dense,Dropout, Activation, Flatten, Convolution1D,Permute,LSTM,MaxPooling1D,Reshape
from keras.optimizers import RMSprop
from keras.models import Sequential
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
        self.learning_rate = 0.01
        self.gamma = 0.6
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
            model.add(Convolution1D(32, 5, border_mode='same', input_shape=(30,8)))
            model.add(MaxPooling1D(pool_size=(30)))
            model.add(Activation('relu'))
            #model.add(Permute((0, 5, 2, 1)))
            model.add(Reshape(target_shape=(30,8)))
            model.add(LSTM(256))
            model.add(Dropout(0.5))
            model.add(LSTM(32,  return_sequences=True))
            model.add(Dropout(0.5))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.action_size))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=self.learning_rate))
        else:
            model.add(LSTM(32,  return_sequences=True,input_shape=(30,8)))
            model.add(Dropout(0.5))
            model.add(LSTM(32,  return_sequences=True))
            model.add(Dropout(0.5))
            model.add(LSTM(32))
            model.add(Dropout(0.5))
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(20, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(10, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))
            model.compile(loss=self.config['loss'], optimizer=Adam(lr=self.learning_rate))
        # Create the model based on the information above
        #model.compile(loss='mean_squared_error', optimizer='adam')
        if os.path.isfile(self.weight_backup):
                    model.load_weights(self.weight_backup)
                    self.exploration_rate = self.exploration_min
        return model
    def save_model(self):
            self.brain.save(self.weight_backup)
    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])
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