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
#from WindPy import *

from matplotlib.font_manager import _rebuild
_rebuild()
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

# 导入数据
price_data = pd.read_pickle('/Users/trevor/Downloads/price.pkl')
indicator_data = pd.read_pickle('/Users/trevor/Downloads/indicator.pkl')
market_index = pd.read_excel('市场底部特征研究.xlsx',sheet_name='沪深300')
M2_data = pd.read_excel('市场底部特征研究.xlsx',sheet_name='M2')


# 合并数据
finance_data = pd.merge(price_data,indicator_data,on=['S_INFO_WINDCODE','TRADE_DT'])

###############################################################
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

###############################################################
#提取PB
stock_PB_data = pd.pivot_table(finance_data,values='S_VAL_PB_NEW',index='S_INFO_WINDCODE',columns='TRADE_DT')
stock_PB_data = stock_PB_data.sort_index(axis=1)
stock_PB_number = stock_PB_data.count(axis=0)
#筛选低PB
low_PB = 1
low_PB_data = stock_PB_data[stock_PB_data<low_PB]
#低PB股数量
low_PB_number = low_PB_data.count(axis=0)
low_PB_percent = low_PB_number/stock_PB_number*100
low_PB_percent.index = pd.DatetimeIndex(low_PB_percent.index)

#############################################################
#处理M2数据
M2_data.set_index('指标名称',inplace=True)
M2_data = M2_data.drop('频率',axis=1)
M2_data = M2_data.T

#计算M2变化率
M2_previous = M2_data[:-1].values
M2_back = M2_data[1:].values
M2_change = M2_back/M2_previous

#提取市值数据
market_value_data = pd.pivot_table(finance_data,values='S_VAL_MV',index='S_INFO_WINDCODE',columns='TRADE_DT')
market_value_data = market_value_data.sort_index(axis=1)
market_value_median = market_value_data.median(axis=0)

market_value_median = market_value_median.reset_index()
market_value_median['MONTH'] = market_value_median['TRADE_DT'].apply(lambda x:x[:6])
market_value_median = pd.merge(market_value_median.groupby(['MONTH'])['TRADE_DT'].max().reset_index(),market_value_median,on=['TRADE_DT'],how='left')
market_value_median.drop(['MONTH_x','MONTH_y'],axis=1,inplace=True)

temp_median = market_value_median.drop(138,axis=0)
temp_M2 = M2_data.reset_index()
temp_median.drop('TRADE_DT',axis=1,inplace=True)
temp_M2.drop('index',axis=1,inplace=True)

M2_MV_ratio = temp_M2.values/temp_median.values
###########################################################

#市场成交额
stock_volume_data = pd.pivot_table(finance_data,values='S_DQ_VOLUME',index='S_INFO_WINDCODE',columns='TRADE_DT')
stock_volume_data = stock_volume_data.sort_index(axis=1)
#市场总成交额
market_volume = stock_volume_data.sum(axis=0)
market_volume.index = pd.DatetimeIndex(market_volume.index)

#个股成交量基准随 M2变化关系
low_volume = 800000
low_volume_data = np.zeros(M2_change.shape)
for i in range(len(M2_change)):
    low_volume = low_volume * M2_change[i]
    low_volume_data[i] = low_volume

#统计个股区间最大跌幅中位数



a = market_index.iloc[0,-1]
b = market_index.iloc[2,-1]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid(False)
ax1.bar(market_volume.index, market_volume.values, width=3,linewidth=0.8,color='yellowgreen',label='全市场成交量',zorder=1)
#ax1.plot(M2_data.index,market_value_median.iloc[:-1,1],color='purple',linewidth=0.8,label='总市值中位数',zorder=3)
ax1.set_ylabel('全市场成交量')
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.grid(False)
#ax2.plot(M2_data.index,M2_MV_ratio,color='orange',linewidth=0.8,label='M2/总市值中位数',zorder=2)
ax2.plot(market_index.columns,market_index.iloc[0,:],color='orange',linewidth=0.8,label='上证综指',zorder=2)
ax2.plot(market_index.columns,market_index.iloc[2,:]*a/b,color='purple',linewidth=0.8,label='深证成指',zorder=3)
ax2.set_ylabel('指数')
ax2.set_ylim(0,7000)
ax2.legend(loc='upper left')
ax2.set_xlabel('时间')
plt.savefig('全市场成交量.jpg',dpi=700)