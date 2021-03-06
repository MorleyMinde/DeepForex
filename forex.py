from agent import Agent
from btgym import BTgymEnv
from monitor import Monitor
from strategy import MyStrategy
from gym import spaces
import numpy as np
import time
import json

class Forex:
    def __init__(self,dir):
        self.dir = dir
        self.config = json.load(open('{}/config.json'.format(dir)))
        self.sample_batch_size = 32
        self.episodes = 10000
        self.time_steps = 30
        self.features = 8
        self.input_shape = (30,4)
        self.monitor = Monitor(dir)
        self.env = BTgymEnv(
            filename=self.config['data'],
            episode_duration={'days': 1, 'hours': 0, 'minutes': 0},
            strategy=MyStrategy,
            start_00=True,
                  start_cash=self.config['capital'],
                  broker_commission=self.config['commission'],
                  fixed_stake=self.config['stake'],
            #drawdown_call=50,
                  state_shape={'raw_state': spaces.Box(low=1, high=2, shape=self.input_shape),'indicator_states': spaces.Box(low=-1, high=10, shape=self.input_shape)},
                  port=5006,
                  data_port=4804,
                  verbose=0,)
        self.state_size        = self.env.observation_space.shape['raw_state'][0] + self.env.observation_space.shape['indicator_states'][0]
        self.action_size       = self.env.action_space.n
        self.agent             = Agent(self.state_size, self.action_size,dir)
        print("Engine:{}".format(self.env.engine.broker.getcash()))
    def run(self):
        #path = dict([(0, 2),(54, 3),(72, 2),(83, 3),(84, 1),(125, 3),(126, 2),(156, 3),(157, 1),(171, 3),(179, 1),(188, 3),(189, 2),(204, 3),(205, 1),(295, 3),(307, 2),(316, 3),(363, 2),(390, 3),(391, 1),(476, 3),(477, 2),(484, 3),(485, 1),(574, 3)])
        try:
            episodes = 0
            for index_episode in range(self.config['episodes'] - self.config['trained_episodes']):
                state = self.env.reset()
                # state = np.reshape(state['raw_state'], [4, self.state_size])
                state = self.getFullState(state)
                #print("State:{}".format(state))
                state = np.reshape(state,(1,self.time_steps,8))
                #print("Re State:{}".format(state))
                # np.concatenate((state['raw_state'],state['indicator_states']),axis=1)
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
                    next_state = np.reshape(next_state,(1,self.time_steps,8))
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
                    print("\x1b[6;30;42m{}\t{} {} \t{} {}\t\t{}\x1b[0m".format(time.strftime("%H:%M:%S"),index + 1,positiveReward,int(final_cash),mapper,message))
                else:
                    if positiveReward > 0:
                        print("{}\t{} {} \t{} {}\t\t{}".format(time.strftime("%H:%M:%S"),index + 1,positiveReward,int(final_cash),mapper,message))
                    #print("{} {} \t{} {}\t\t{}".format(index + 1,positiveReward,int(final_cash),mapper,message))
                self.agent.replay(self.sample_batch_size)
        finally:
            self.config['trained_episodes'] += episodes
            self.agent.save_model()
            self.monitor.close()
            with open('{}/config.json'.format(self.dir), 'w') as outfile:
               json.dump(self.config, outfile)

    def getFullState(self,state):
        return np.concatenate((state['raw_state'],state['indicator_states']),axis=1)

    def testInitial(self):
        with open("log.txt", "a") as myfile:
            env = BTgymEnv(
                filename='./btgym/examples/data/DAT_ASCII_EURUSD_M1_2016.csv',
                start_cash=100,
                broker_commission=0.0001,
                leverage=10.0,
                fixed_stake=10)

            done = False

            o = env.reset()

            while not done:
                action = env.action_space.sample()
                obs, reward, done, info = env.step(action)
                myfile.write('action: {},reward: {},info: {}\n'.format(action, reward, info))

        env.close()
    def test(self):
        state = self.env.reset()
        # state = np.reshape(state['raw_state'], [4, self.state_size])
        state = self.getFullState(state)
        state = np.reshape(state,(1,self.time_steps,8))
        done = False
        index = 0
        final_cash = 0
        message = ""
        while not done:
            #self.env.render()
            action = self.agent.act(state)
            next_state, reward, done, info = self.env.step(action)
            final_cash = info[0]['broker_cash']
            message = info[0]['broker_message']
            # next_state = np.reshape(next_state, [1, self.state_size])
            # next_state = np.array(next_state['raw_state'])
            # print("Shape1 :{} Shape2 :{}".format(next_state['raw_state'].shape,next_state['indicator_states'].shape))
            next_state = self.getFullState(next_state)
            next_state = np.reshape(next_state,(1,self.time_steps,8))
            state = next_state
            index += 1
        print("\x1b[6;30;42m{}\t{} \t{} \t\t{}\x1b[0m".format(time.strftime("%H:%M:%S"),index + 1,final_cash,message))
        # print("Final Amount:{}".format())