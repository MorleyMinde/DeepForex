from btgym import BTgymEnv, BTgymBaseStrategy, BTgymDataset
import numpy as np
import backtrader.indicators as btind
import datetime

class MyStrategy(BTgymBaseStrategy):
    def __init__(self, **kwargs):
        self.dim_time = self.p.state_shape[list(self.p.state_shape.keys())[0]].shape[0]
        BTgymBaseStrategy.__init__(self,**kwargs)
    """
    Example subclass of BTgym inner computation startegy,
    overrides default get_state() and get_reward() methods.
    """
    def set_datalines(self):
        self.indicators = []
        period = self.dim_time
        self.indicators.append(btind.ExponentialMovingAverage(
            self.datas[0],
            period=period
        ))
        self.indicators.append(btind.MACD(
            self.datas[0]
        ))
        self.indicators.append(btind.WilliamsAD(
            self.datas[0]
        ))
        self.indicators.append(btind.BollingerBands(
            self.datas[0],
            period=period
        ))
    def _get_raw_state(self):
        """
        Default state observation composer.

        Returns:
             and updates time-embedded environment state observation as [n,4] numpy matrix, where:
                4 - number of signal features  == state_shape[1],
                n - time-embedding length  == state_shape[0] == <set by user>.

        Note:
            `self.raw_state` is used to render environment `human` mode and should not be modified.

        """
        self.raw_state = np.row_stack(
            (
                np.frombuffer(self.data.open.get(size=self.dim_time)),
                np.frombuffer(self.data.high.get(size=self.dim_time)),
                np.frombuffer(self.data.low.get(size=self.dim_time)),
                np.frombuffer(self.data.close.get(size=self.dim_time)),
            )
        ).T

        return self.raw_state
    def get_state(self):
        X = self.raw_state
        self.state['raw_state'] = X
        self.state.pop('indicator_states', None)
        if True:
            for indicator in self.indicators:
                if 'indicator_states' not in self.state:
                    self.state['indicator_states'] = np.row_stack((np.frombuffer(indicator.get(size=self.dim_time)),)).T
                else:
                    self.state['indicator_states'] = np.concatenate((self.state['indicator_states'],np.row_stack((np.frombuffer(indicator.get(size=self.dim_time)),)).T),axis=1)
        self.state['indicator_states'][np.isnan(self.state['indicator_states'])] = 1
        self.state['indicator_states'][:,3] = self.state['indicator_states'][:,3] /100
        return self.state


import numpy as np
# The above could be sent to an independent module
import backtrader as bt
from agent import Agent

HOLD = 0
BUY = 1
SELL = 2
STOP = 3
class DeepLearningStrategy(bt.Strategy):
    params = dict(
        smaperiod=5,
        trade=False,
        stake=10,
        exectype=bt.Order.Market,
        stopafter=0,
        valid=None,
        cancel=0,
        donotsell=False,
        stoptrail=False,
        stoptraillimit=False,
        trailamount=None,
        trailpercent=None,
        limitoffset=None,
        oca=False,
        bracket=False,
    )

    def __init__(self):
        # To control operation entries
        self.orderid = list()
        self.order = None

        self.counttostop = 0
        self.datastatus = 0

        self.state = np.empty([4, 8])
        self.stateFilled = 0
        self.agent = Agent(8, 4)
        # Create SMA on 2nd data
        self.ema = bt.indicators.MovAv.EMA(self.data, period=self.p.smaperiod)
        #self.ema = bt.indicators.MovAv.SMA(self.data, period=self.p.smaperiod)
        self.macd = btind.MACD(self.data)
        self.williamad = btind.WilliamsAD(self.data)
        self.bollinger = btind.BollingerBands(self.data,period=self.p.smaperiod )

        self.currentOrder = None
        print('--------------------------------------------------')
        print('Strategy Created')
        print('--------------------------------------------------')

    def notify_data(self, data, status, *args, **kwargs):
        print('*' * 5, 'DATA NOTIF:', data._getstatusname(status),self.p.stopafter, args)
        if status == data.LIVE:
            self.counttostop = self.p.stopafter
            self.datastatus = 1

    def notify_store(self, msg, *args, **kwargs):
        print('*' * 5, 'STORE NOTIF:', msg)

    def notify_order(self, order):
        if order.status in [order.Completed, order.Cancelled, order.Rejected]:
            self.order = None

        print('-' * 50, 'ORDER BEGIN', datetime.datetime.now())
        print(order)
        print('-' * 50, 'ORDER END')

    def notify_trade(self, trade):
        print('-' * 50, 'TRADE BEGIN', datetime.datetime.now())
        print(trade)
        print('-' * 50, 'TRADE END')

    def prenext(self):
        self.next(frompre=True)

    def next(self, frompre=False):
        for i in range(0,len(self.state)):
            try:
                self.state[i] = self.state[i +1]
            except IndexError:
                self.state[i][0] = self.data.open[0]
                self.state[i][1] = self.data.high[0]
                self.state[i][2] = self.data.low[0]
                self.state[i][3] = self.data.close[0]
                self.state[i][4] = self.ema[0]
                self.state[i][5] = self.macd[0]
                self.state[i][6] = self.williamad[0]
                self.state[i][7] = self.bollinger[0]
        if self.stateFilled == self.p.smaperiod:
            if self.datastatus == 1:
                exectype = self.p.exectype if not self.p.oca else bt.Order.Limit
                close = self.data0.close[0]
                price = round(close * 0.90, 2)
                action = self.agent.act(self.state)
                if (self.currentOrder != action and action != HOLD) or action == STOP:
                    for i in range(len(self.orderid)):
                        self.cancel(self.orderid[i])
                        self.close(self.orderid[i])
                        del self.orderid[i]
                    self.order = None
                    print("Cancel Order")
                if action == BUY:
                    self.order = self.buy(size=self.p.stake,
                                  exectype=exectype,
                                  price=price,
                                  valid=self.p.valid,
                                  transmit=not self.p.bracket)
                    print("Buy order:{}".format(self.broker.getcash()))
                    self.orderid.append(self.order)
                elif action == SELL:
                    self.order = self.sell(size=self.p.stake,
                                  exectype=exectype,
                                  price=price,
                                  valid=self.p.valid,
                                  transmit=not self.p.bracket)
                    print("Sell order:{}".format(self.broker.getcash()))
                    self.orderid.append(self.order)
                if not self.currentOrder:
                    self.currentOrder = action
        else:
            self.stateFilled += 1

    def start(self):
        if self.data0.contractdetails is not None:
            print('Timezone from ContractDetails: {}'.format(
                  self.data0.contractdetails.m_timeZoneId))

        header = ['Datetime', 'Open', 'High', 'Low', 'Close', 'Volume',
                  'OpenInterest', 'SMA']
        print(', '.join(header))

        self.done = False
