from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import backtrader as bt
import backtrader.analyzers as btanalyzers
import mlmod
import pandas as pd
import datacapture as dc
import modes
from datetime import timedelta as de
from datetime import datetime as dt
import time


class MyStrategy(bt.Strategy):
    def __init__(self):
        self.dclose=self.datas[0].close
        self.dopen=self.datas[0].open
        self.dhigh=self.datas[0].high
        self.dlow=self.datas[0].low
        self.dvol=self.datas[0].volume

        self.printlog=True
        self.order= None
        self.buyprice=None
        self.buycomm=None
        self.endog=None
        self.exog=None
        self.posa = None
        
        self.prdf1 = pd.DataFrame({'o':self.datas[1].open.array
                                ,'h':self.datas[1].high.array
                                ,'l':self.datas[1].low.array
                                ,'c':self.datas[1].close.array
                                ,'v':self.datas[1].volume.array
                                })
        self.proc = modes.dump(data=self.prdf1,asset='ETH-USD',hrm='even',m=1,F=368896)
        self.exog = self.proc.pos
        self.endog = self.proc.loc[:,'x1':'Spd']
        self.clf = mlmod.MLP(self.endog,self.exog,asset='ETH-USD')

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function fot this strategy'''
        if self.printlog:
            dt = dt or self.datas[0].datetime.datetime(0)
            print('\n','%s, %s' % (dt.isoformat(), txt))
            time.sleep(0)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm: %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            elif order.issell():  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm: %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS: %.2f, NET: %.2f' %
                    (trade.pnl, trade.pnlcomm))

    def next(self):
        self.log(f'TICKER, Close: {self.dclose[0]}, Volume: {self.dvol[0]}, Inpos: {self.position.upopened}')

        self.prdf1 = self.prdf1.shift(-1)
        self.prdf1.iloc[-1] = pd.DataFrame(index=[0],data={'o':self.data0.open[0]
                                ,'h':self.data0.high[0]
                                ,'l':self.data0.low[0]
                                ,'c':self.data0.close[0]
                                ,'v':self.data0.volume[0]
                                })
        self.prdf1.reset_index(drop=True,inplace=True)
        self.proc = modes.dump(data=self.prdf1,asset='ETH-USD',hrm='even',m=1,F=368896)
        self.endog = self.proc.loc[:,'x1':'Spd']
        self.exog = self.proc.pos
        self.clf=self.clf.partial_fit(self.endog,self.exog)
        self.posa = self.clf.predict(self.endog)

        if self.order:
            return

        # Check if we are in the market
        if not self.position:
            # self.order = self.buy(size=1)
            # Not yet ... we MIGHT BUY if ...
            if int(self.posa[-1])==1:

                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy(data=self.data0,size=1)

        else:
            if int(self.posa[-1])==0:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell(data=self.data0,size=1)
        
#############################################################################
#Instantiate Cerebro engine
cerebro = bt.Cerebro()

asset = 'ETH-USD'
start=dt.now().replace(second=0,microsecond=0)-de(days=365)
stop=dt.now().replace(second=0,microsecond=0)-de(seconds=3600)
hrm='even'
interval='1minute'
F=368896
mode='dump'
obv=['v','c']
m=1

# df = dc.get_data_span(asset=asset,start=start,stop=stop,interval=interval,mode=mode)

df  = pd.read_csv('test_data.csv')
df.dt=pd.to_datetime(df.dt,format='%Y-%m-%d %H:%M:%S')
df.set_index('dt',drop=True,inplace=True)

df=df.loc[df.index[-1]-de(days=30*4):]

idx  = 60*3
testdta = df.iloc[-idx:]
traindta = df.iloc[:-idx]

data = bt.feeds.PandasData(dataname=testdta
                            ,datetime=None
                            ,open = 0
                            ,high=1
                            ,low=2
                            ,close=3
                            ,volume=4
                            ,openinterest=-1
                            )
data1 = bt.feeds.PandasData(dataname=traindta
                            ,datetime=None
                            ,open = 0
                            ,high=1
                            ,low=2
                            ,close=3
                            ,volume=4
                            ,openinterest=-1
                            )

cerebro.adddata(data)

cerebro.adddata(data1)

#Add strategy to Cerebro
cerebro.addstrategy(MyStrategy)

# Set our desired cash start
cerebro.broker.setcash(100000.0)

# Analyzers
cerebro.addanalyzer(btanalyzers.TradeAnalyzer,   _name='mypos')

# Observers
cerebro.addobserver(bt.observers.Broker)
cerebro.addobserver(bt.observers.Trades)
cerebro.addobserver(bt.observers.BuySell,barplot=True,bardist=0.015)

# Add a FixedSize sizer according to the stake
cerebro.addsizer(bt.sizers.SizerFix, stake=20)

# Set the commission
cerebro.broker.setcommission(commission=0.00)

print(f'\nBACKTESTING {asset} FOR DATES {start} to {stop} @ {interval}\n')

# Print out the starting conditions
print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

# Run over everything
thestrats = cerebro.run(writer=True
                        ,tradehistory=True
                        ,stdstats=False)
thestrat = thestrats[0]

# Print out the final result
print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
print('Trade Analysis:\n' ,thestrat.analyzers.mypos.get_analysis())

cerebro.plot()



