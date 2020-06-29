# Author: Joshua Jackson
# This script will be to plot and run trades on historical data
# using the sentiment analysis strategy
#
# This script was largely based on a script found on a medium post
# https://towardsdatascience.com/https-towardsdatascience-com-algorithmic-trading-using-sentiment-analysis-on-news-articles-83db77966704
# Written by Jason Yip

from __future__ import (absolute_import, division, print_function,unicode_literals)

# %matplotlib inline
# import warnings
# warnings.filterwarnings('ignore')

import backtrader as bt
import backtrader.indicators as btind
from backtrader import plot
import datetime
import os.path
import sys

class Sentiment(bt.Indicator):
    lines = ('sentiment',)
    plotinfo = dict(
        plotymargin=0.15,
        plothlines=[0],
        plotyticks=[1.0, 0, -1.0])
    
    def next(self):
        self.date = self.data.datetime
        date = bt.num2date(self.date[0]).date()
        prev_sentiment = self.sentiment
        if date in date_sentiment:
            self.sentiment = date_sentiment[date]
        else:
            self.sentiment = 0.0
        self.lines.sentiment[0] = self.sentiment

class SentimentStrat(bt.Strategy):
    params = (
        ('period', 10),
        ('printlog', True),
    )

    def log(self, txt, dt=None, doprint=False):
        ''' Logging function for this strategy'''
        if self.params.printlog or doprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close
        # Keep track of pending orders
        self.order = None
        self.buyprice = None
        self.buycomm = None
        self.sma = bt.indicators.SimpleMovingAverage(
            self.datas[0], period=self.params.period)
        self.date = self.data.datetime
        self.sentiment = None
        Sentiment(self.data)
        
    def notify_order(self, order):
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
            
        # Write down: no pending order
        self.order = None
        
    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))
    
    ### Main Strat ###
    def next(self):
        # log closing price of the series from the reference
        if self.sentiment:
            self.log('Close, %.2f - Sentiment, %.2f' % (self.dataclose[0], self.sentiment))
        
        date = bt.num2date(self.date[0]).date()
        prev_sentiment = self.sentiment
        if date in date_sentiment:
            self.sentiment = date_sentiment[date]
        
        # Check if an order is pending. if yes, we cannot send a 2nd one
        if self.order:
            return
        # If not in the market and previous sentiment not none
        if not self.position and prev_sentiment:
            # buy if current close more than sma AND sentiment increased by >= 0.5
#             if self.dataclose[0] > self.sma[0] and self.sentiment - prev_sentiment >= 0.25:
            if self.sentiment - prev_sentiment >= 0.25:
                self.log('BUY CREATE, %.2f - SENTIMENT %.2f' % (self.dataclose[0], (self.sentiment - prev_sentiment)))
                self.order = self.buy()
                
        # Already in the market and previous sentiment not none
        elif prev_sentiment:
            # sell if current close less than sma AND sentiment decreased by >= 0.5
#             if self.dataclose[0] < self.sma[0] and self.sentiment - prev_sentiment <= -0.5:
            if self.sentiment-prev_sentiment <= -0.25:
                self.log('SELL CREATE, %.2f - SENTIMENT %.2f' % (self.dataclose[0], (self.sentiment - prev_sentiment)))
                self.order = self.sell()

    def stop(self):
        self.log('(MA Period %2d) Ending Value %.2f' % (self.params.period, self.broker.getvalue()), doprint=True)

class plotCerebro:

    def saveplots(cerebro, numfigs=1, iplot=True, start=None, end=None, width=16, height=9, dpi=300, tight=True, use=None, file_path = '', **kwargs):
        if cerebro.p.oldsync:
            plotter = plot.Plot_OldSync(**kwargs)
        else:
            plotter = plot.Plot(**kwargs)

        figs = []
        for stratlist in cerebro.runstrats:
            for si, strat in enumerate(stratlist):
                rfig = plotter.plot(strat, figid=si * 100,
                                    numfigs=numfigs, iplot=iplot,
                                    start=start, end=end, use=use)
                figs.append(rfig)

        for fig in figs:
            for f in fig:
                f.savefig(file_path, bbox_inches='tight')
        return figs



### main to run class
#if __name__ == '__main__':
#    cerebro = bt.Cerebro(stdstats=False)
#    
#    # Strategy
#    cerebro.addstrategy(SentimentStrat)
#
#    # Data Feed
#    data = bt.feeds.PandasData(dataname=stock_df)
#    
#    cerebro.adddata(data)
#
#    cerebro.broker.setcash(100000.0)
#    cerebro.addsizer(bt.sizers.FixedSize, stake=100)
#    cerebro.broker.setcommission(commission=0.001)
#    
#    #print starting value of portfolio
#    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
#    
#    cerebro.addobserver(bt.observers.Trades)
#    cerebro.addobserver(bt.observers.BuySell)
#    cerebro.run()
#    
#    #set plot paramaters
#    plt.rcParams['figure.figsize']=[18, 16]
#    plt.rcParams['figure.dpi']=200
#    plt.rcParams['figure.facecolor']='w'
#    plt.rcParams['figure.edgecolor']='k'
#    
#    #print final protfolio value based on data given
#    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
#    
#    dateTimeObj = datetime.datetime.now()
#    timestampStr = dateTimeObj.strftime("%d-%b-%Y")
#    
#    #save plot to file
#    plotCerebro().saveplots(cerebro, file_path = f'backtraded-plot-{timestamp}.png')
#    
#    #plot it
#    cerebro.plot()
    
    
