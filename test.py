# from forex import Forex
# from shutil import copyfile
# import sys
#
# print(sys.argv[1])
# if __name__ == "__main__":
#
#     copyfile("./training/{}/weights.h5".format(sys.argv[1]), "./training/{}/test/weights.h5".format(sys.argv[1]))
#     forex = Forex("./training/{}/test/".format(sys.argv[1]))
#     forex.test()

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])
import numpy as np

# Import the backtrader platform
import backtrader as bt
import agent

class OandaCSVData(bt.feeds.GenericCSVData):
    params = (
        ('nullvalue', float('NaN')),
        ('dtformat', '%Y%m%d %H%M%S'),
        ('datetime', 0),
        ('time', -1),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('openinterest', -1),
    )
import strategy
agent = agent.Agent(30, 4,10,"/home/vincent/PycharmProjects/deep-forex/training/{}/".format(sys.argv[1]))
# Create a Stratey
class DeepStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' Logging function for this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.state = np.zeros((30,10))
        self.steps = 0
        self.open = self.datas[0].open
        self.high = self.datas[0].high
        self.low = self.datas[0].low
        self.close = self.datas[0].close
        self.volume = self.datas[0].volume
        self.ema = bt.indicators.ExponentialMovingAverage(
            self.datas[0],
            period=15
        )
        self.macd = bt.indicators.MACDHisto(
            self.datas[0]
        )
        self.rsi = bt.indicators.RSI(
            self.datas[0],
            period=15,safediv=True
        )
        self.stochastic = bt.indicators.StochasticFast(
            self.datas[0],safediv=True
        )
        self.rv = strategy.RelativeVolume(
            self.datas[0],period=30
        )
    def getNext(self,new_data):
        for index in range(self.state.shape[0]):
            if self.state.shape[0] == index + 1:
                self.state[index] = new_data
            else:
                self.state[index] = self.state[index + 1]
    def notify_order(self, order):
        print("Order:{} {} {}".format(order.status,order.Submitted, order.Accepted))
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))
    def next(self):
        self.steps += 1
        self.getNext([self.open[0],self.high[0],self.low[0],self.close[0],self.volume[0],self.ema[0],self.macd[0],self.rsi[0],self.stochastic[0],self.rv[0]])
        # Simply log the closing price of the series from the reference
        # self.log('EMA:{},macd:{},rsi:{},stoc:{},rv:{},close:{}'.format(self.ema[0],self.macd[0],self.rsi[0],self.stochastic[0],self.rv[0],self.volume[0]))
        action = agent.act(self.state)

        # Check if we are in the market
        if action == 1:
            if self.order:
                if self.order.issell():
                    self.log('Buy')
                    self.close()
                    self.order = self.buy()
            else:
                self.log('Buy')
                self.order = self.buy()
        elif action == 2:
            if self.order:
                if self.order.isbuy():
                    self.log('Sell')
                    self.close()
                    self.order = self.sell()
            else:
                self.log('Sell')
                self.order = self.sell()
        elif action == 3 and self.order:
            self.log('Close')
            self.close()
            # self.order = self.close()


if __name__ == '__main__':
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(DeepStrategy)

    # Datas are in a subfolder of the samples. Need to find where the script is
    # because it could have been called from anywhere
    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, '/home/vincent/PycharmProjects/deep-forex/data/DAT_ASCII_EURUSD_M1_20170801.csv')

    # Create a Data Feed
    data = OandaCSVData(
        dataname=datapath,
        # Do not pass values before this date
        # fromdate=datetime.datetime(2018, 1, 1),
        # Do not pass values before this date
        # todate=datetime.datetime(2018, 1, 31),
        # Do not pass values after this date
        )

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100.0)

    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=10)

    # Set the commission - 0.1% ... divide by 100 to remove the %
    cerebro.broker.setcommission(commission=0.001)

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    # Print out the final result
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # cerebro.plot()