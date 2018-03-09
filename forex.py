from agent import Agent
from btgym import BTgymEnv,BTgymDataset
from btgym.datafeed.base import BTgymBaseData
import backtrader.feeds as btfeeds
from monitor import Monitor
from strategy import MyStrategy
from gym import spaces
import numpy as np
import time
import json
from btdata import BTgymExtendDataset

class Forex:
    def __init__(self,dir):
        self.dir = dir
        self.config = json.load(open('{}/config.json'.format(dir)))
        self.sample_batch_size = 32
        self.episodes = 10000
        self.time_steps = 30
        self.features = 8
        self.input_shape = (30,6)
        self.monitor = Monitor(dir)
        params = dict(
            # CSV to Pandas params.
            sep=';',
            header=0,
            index_col=0,
            parse_dates=True,
            names=['open', 'high', 'low', 'close', 'volume'],

            # Pandas to BT.feeds params:
            timeframe=1,  # 1 minute.
            datetime=0,
            open=1,
            high=2,
            low=3,
            close=4,
            volume=5,
            openinterest=-1,

            start_weekdays=[0, 1, 2, 3, ],  # Only weekdays from the list will be used for episode start.
            start_00=True,  # Episode start time will be set to first record of the day (usually 00:00).

            )
        self.env = BTgymEnv(
            # dataset=BTgymDataset(
            #     filename=self.config['data'],
            #     **params
            # ),
            filename=self.config['data'],
            episode_duration={'days': 1, 'hours': 0, 'minutes': 0},
            strategy=MyStrategy,
            start_00=True,
                  start_cash=self.config['capital'],
                  broker_commission=self.config['commission'],
                  fixed_stake=self.config['stake'],
            drawdown_call=10,
                  state_shape=dict({
                      'raw_state': spaces.Box(shape=(30,4),low=0, high=2),'indicator_states': spaces.Box(low=-1, high=100, shape=self.input_shape)
                  }),
                  port=self.config['port'],
                  data_port=self.config['data_port'],
                  verbose=0,)
        self.state_size        = self.env.observation_space.shape['raw_state'][0] + self.env.observation_space.shape['indicator_states'][0]
        self.action_size       = self.env.action_space.n
        self.agent             = Agent(self.state_size, self.action_size,10,dir)
        print("Engine:{} {}".format(self.env.engine.broker.getcash(),self.state_size))
    def run(self):
        try:
            episodes = 0
            for index_episode in range(self.config['episodes'] - self.config['trained_episodes']):
                state = self.env.reset()
                state = self.getFullState(state)
                done = False
                index = 0
                final_cash = 0
                message = ""

                count = 0
                negativeReward = 0
                positiveReward = 0
                mapper = {
                    0:0,
                    1:0,
                    2:0,
                    3:0,
                }
                while not done:
                    # self.env.render()
                    action = self.agent.act(state)
                    next_state, reward, done, info = self.env.step(action)
                    mapper[action] += 1
                    if reward > 0:
                        positiveReward += reward
                    else:
                       negativeReward += reward
                    # next_state = np.reshape(next_state, [1, self.state_size])
                    # next_state = np.array(next_state['raw_state'])
                    # print("Shape1 :{} Shape2 :{}".format(next_state['raw_state'].shape,next_state['indicator_states'].shape))

                    final_cash = info[0]['broker_cash']
                    message = info[0]['broker_message']
                    next_state = self.getFullState(next_state)
                    #print("Action:{} Reward:{} Done:{}".format(action, reward,done))
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                    self.config['steps'] += info[0]['step']
                    self.monitor.logstep({"reward":reward,"drawdown":info[0]['drawdown'],'broker_value':info[0]['broker_value'],"steps":self.config['steps']})
                    if self.config['steps'] % 100 == 0:
                        self.monitor.logimage(feed_dict={'human': self.env.render('human')[None,:],},global_step=self.config['steps'],)
                    # print('action: {},reward: {},info: {}\n'.format(action, reward, info))
                episodes += 1
                episode_stat = self.env.get_stat()
                self.monitor.logepisode({"reward":reward,"cpu_time_sec":episode_stat['runtime'].total_seconds(),"global_step":self.config['trained_episodes'] + episodes,'broker_value':info[0]['broker_value'],"episode":self.env.render('episode')[None,:]})
                if "CLOSE, END OF DATA" == message:
                    if positiveReward > 0:
                        print("\x1b[6;30;42m{} \t{}\t{}\t {:.4f} \t{:.3f}\t\t{} {:.3f}\x1b[0m".format(self.config['trained_episodes'],time.strftime("%H:%M:%S"),index + 1,positiveReward,final_cash,message,negativeReward))
                    else:
                        print("\x1b[6;30;43m{} \t{}\t{}\t {:.4f} \t{:.3f}\t\t{} {:.3f}\x1b[0m".format(self.config['trained_episodes'],time.strftime("%H:%M:%S"),index + 1,positiveReward,final_cash,message,negativeReward))
                else:
                    if positiveReward > 0:
                        print("\x1b[6;30;43m{} \t{}\t{}\t {:.4f} \t{:.3f}\t\t{}\x1b[0m".format(self.config['trained_episodes'],time.strftime("%H:%M:%S"),index + 1,int(positiveReward * 10),int(final_cash),message))
                    else:
                        print("{} \t{}\t{}\t {:.4f} \t{:.3f}\t\t{}".format(self.config['trained_episodes'],time.strftime("%H:%M:%S"),index + 1,int(positiveReward * 10),int(final_cash),message))
                self.agent.replay(self.sample_batch_size)
                self.agent.save_model()
                self.config['trained_episodes'] += 1
                with open('{}/config.json'.format(self.dir), 'w') as outfile:
                    json.dump(self.config, outfile)
        finally:
            self.config['trained_episodes'] += episodes
            self.monitor.close()
            with open('{}/config.json'.format(self.dir), 'w') as outfile:
               json.dump(self.config, outfile)

    def getFullState(self,state):
        if self.config['network'] == "CNNRNN":
            return self.convertTo3d(np.concatenate((state['raw_state'],state['indicator_states']),axis=1))
        elif self.config['network'] == "CNNRNN2":
            return np.concatenate((state['raw_state'],state['indicator_states']),axis=1)
        else:
            return np.reshape(np.concatenate((state['raw_state'],state['indicator_states']),axis=1),(1,30,8))
    def convertTo3d(self,state):
        for i in range(state.shape[1]):
            state[:,i-1] = (state[:,i-1]-min(state[:,i-1]))/(max(state[:,i-1])-min(state[:,i-1])) * 29.0
        state = state.astype(int)
        state[state < 0] = 0
        new = np.zeros((30,30,8), dtype=int)
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                new[i - 1][state[i][j-1]][j] = 1
        return new

    def test(self):
        for index_episode in range(5):
                state = self.env.reset()
                state = self.getFullState(state)
                done = False
                index = 0
                final_cash = 0
                message = ""

                count = 0
                negativeReward = 0
                positiveReward = 0
                mapper = {
                    0:0,
                    1:0,
                    2:0,
                    3:0,
                }
                while not done:
                    # self.env.render()
                    action = self.agent.act(state)
                    next_state, reward, done, info = self.env.step(action)
                    mapper[action] += 1
                    if reward > 0:
                        positiveReward += reward
                    else:
                       negativeReward += reward
                    # next_state = np.reshape(next_state, [1, self.state_size])
                    # next_state = np.array(next_state['raw_state'])
                    # print("Shape1 :{} Shape2 :{}".format(next_state['raw_state'].shape,next_state['indicator_states'].shape))

                    final_cash = info[0]['broker_cash']
                    message = info[0]['broker_message']
                    next_state = self.getFullState(next_state)

                    state = next_state
                    index += 1
                episode_stat = self.env.get_stat()
                if "CLOSE, END OF DATA" == message:
                    if positiveReward > 0:
                        print("\x1b[6;30;42m{} \t{}\t{}\t {:.4f} \t{:.3f}\t\t{} {:.3f}\x1b[0m".format(self.config['trained_episodes'],time.strftime("%H:%M:%S"),index + 1,positiveReward,final_cash,message,negativeReward))
                    else:
                        print("\x1b[6;30;43m{} \t{}\t{}\t {:.4f} \t{:.3f}\t\t{} {:.3f}\x1b[0m".format(self.config['trained_episodes'],time.strftime("%H:%M:%S"),index + 1,positiveReward,final_cash,message,negativeReward))
                else:
                    if positiveReward > 0:
                        print("\x1b[6;30;43m{} \t{}\t{}\t {} \t{}\t\t{}\x1b[0m".format(self.config['trained_episodes'],time.strftime("%H:%M:%S"),index + 1,int(positiveReward * 10),int(final_cash),message))
                    else:
                        print("{} \t{}\t{}\t {} \t{}\t\t{}".format(self.config['trained_episodes'],time.strftime("%H:%M:%S"),index + 1,int(positiveReward * 10),int(final_cash),message))