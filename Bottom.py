# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:09:25 2018

@author: Zhehao Li
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from sklearn import linear_model
from WindPy import *
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['FangSong'] #
mpl.rcParams['axes.unicode_minus'] = False 

# 导入数据
price_data = pd.read_pickle('price.pkl')
indicator_data = pd.read_pickle('indicator.pkl')
market_index = pd.read_excel('市场底部特征研究.xlsx',sheetname='沪深300')

# 合并数据
finance_data = pd.merge(price_data,indicator_data,on=['S_INFO_WINDCODE','TRADE_DT'])


#提取日收盘价
stock_close_data = pd.pivot_table(finance_data,values='S_DQ_CLOSE',index='S_INFO_WINDCODE',columns='TRADE_DT')                   
stock_close_data = stock_close_data.sort_index(axis=1)
stock_close_number = stock_close_data.count(axis=0)
#筛选出低价股
low_price=2
low_price_data = stock_close_data[stock_close_data<low_price]
#低价股数量
low_price_number = low_price_data.count(axis=0)
low_price_percent = low_price_number/stock_close_number*100
low_price_percent.index = pd.DatetimeIndex(low_price_percent.index)


#提取PB
stock_PB = pd.pivot_table(finance_data,values='S_VAL_PB_NEW',index='S_INFO_WINDCODE',columns='TRADE_DT')
stock_PB = stock_PB.sort_index(axis=1)
stock_PB_number = stock_PB.count(axis=0)
#筛选低PB
low_PB = 1
low_PB_data = stock_PB[stock_PB<low_PB]
#低PB股数量
low_PB_number = low_PB_data.count(axis=0)
low_PB_percent = low_PB_number/stock_PB_number*100
low_PB_percent.index = pd.DatetimeIndex(low_PB_percent.index)


a = market_index.iloc[0,-1]
b = market_index.iloc[2,-1]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(market_index.columns,market_index.iloc[0,:],color='orange',linewidth=0.8,label='上证综指',zorder=3)
ax1.plot(market_index.columns,market_index.iloc[2,:]*a/b,color='purple',linewidth=0.8,label='深证成指',zorder=2)
ax1.set_ylabel('上证指数')
ax1.set_ylim(0,7000)
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.bar(low_price_percent.index, low_price_percent.values, width=3,linewidth=0.8,color='yellowgreen',label='低价股',zorder=1)
ax2.set_ylabel('低价股比例')
ax2.legend(loc='upper right')
ax2.set_xlabel('时间')
plt.savefig('低价股比例',dpi=700)
