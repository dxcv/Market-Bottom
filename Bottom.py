
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

from matplotlib.font_manager import _rebuild
_rebuild()
plt.rcParams['font.sans-serif'] = ['FangSong'] 
plt.rcParams['axes.unicode_minus'] = False 

from WindPy import * 
w.start()
nationbonds = w.edb("M0325687", "2007-01-04", "2018-07-30")
nationbonds.
w.close()


# 导入数据
price_data = pd.read_pickle('price.pkl')
indicator_data = pd.read_pickle('indicator.pkl')
market_index = pd.read_excel('市场底部特征研究.xlsx',sheet_name='沪深300')
M2_data = pd.read_excel('M2指标.xlsx')
IPO_date = pd.read_excel('A股上市时间.xlsx')
amount_data = pd.read_csv('eodprices.csv')

###############################################################################
'''
0. 准备工作
'''
# 合并数据
finance_data = pd.merge(price_data,indicator_data,on=['S_INFO_WINDCODE','TRADE_DT'])
amount_data['TRADE_DT'].apply(lambda x:pd.to_datetime(str(x),format='%Y%m%d'))

#提取日收盘价
stock_close_data = pd.pivot_table(finance_data,values='S_DQ_CLOSE',index='S_INFO_WINDCODE',columns='TRADE_DT')                   
stock_close_data = stock_close_data.sort_index(axis=1)
stock_close_number = stock_close_data.count(axis=0)


###############################################################################
'''
1.低价股比例
'''
#筛选出低价股
low_price=2
low_price_data = stock_close_data[stock_close_data<low_price]
#低价股数量
low_price_number = low_price_data.count(axis=0)
low_price_percent = low_price_number/stock_close_number
low_price_percent.index = pd.DatetimeIndex(low_price_percent.index)

###############################################################################
'''
2.破净股比例
'''
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


###############################################################################
'''
3.M2/总市值中位数
'''
#处理M2数据
M2_data.set_index('指标名称',inplace=True)
M2_data = M2_data.drop('频率',axis=1)
M2_data = M2_data.T

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
###############################################################################
'''
4.PE中位数&十年国债收益率倒数
'''

#提取PE
stock_PE_data = pd.pivot_table(finance_data,values='S_VAL_PE_TTM',index='S_INFO_WINDCODE',columns='TRADE_DT')
stock_PE_data = stock_PE_data.sort_index(axis=1)
stock_PE_median = stock_PE_data.median(axis=0)
stock_PE_median.index = pd.DatetimeIndex(stock_PE_median.index)

#十年国债收益率
nationbonds_interest = pd.Series(np.array(nationbonds.Data).reshape(-1),index=nationbonds.Times)
nationbonds_interest.index = pd.DatetimeIndex(nationbonds_interest.index)

###############################################################################
'''
5.全市场成交额（市场人气）
'''
#所有股票成交额
stock_amount_data = pd.pivot_table(amount_data,values='S_DQ_AMOUNT',index='S_INFO_WINDCODE',columns='TRADE_DT')
stock_amount_data = stock_amount_data.sort_index(axis=1)
#市场总成交额
total_amount = stock_amount_data.sum(axis=0)

###############################################################################
'''
6.个股成交额（个股流动性）
'''
#筛选低成交额个股
low_volume = 10000 #(千元)
low_volume_stock = stock_volume_data[stock_volume_data<low_volume]
low_volume_number = low_volume_stock.count(axis=0)
low_volume_percent = low_volume_number/stock_close_number
low_volume_percent.index = pd.DatetimeIndex(low_volume_percent.index)

###############################################################################
'''
7.个股区间最大跌幅中位数（月区间）
'''






###############################################################################
'''
8.次新股(上市四个月内)破发率（市场人气）
'''

#筛选出时间横截面股票
for day in stock_close_data.columns:
    





###############################################################################

c = market_index.iloc[0,-1]
d = market_index.iloc[2,-1]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid(False)
ax1.bar(low_volume_percent.index, low_volume_percent, width=3,linewidth=0.8,color='yellowgreen',label='低成交额比例',zorder=1)
#.plot(M2_data.index, M2_data*a/b, color='purple',linewidth=0.8,label='低成交额基准线',zorder=2)
ax1.set_ylabel('低成交额比例')
#ax1.set_ylim(0,130)
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.grid(True)
#ax2.plot(nationbonds_interest.index,1/nationbonds_interest,color='orange',linewidth=0.8,label='十年国债收益率倒数',zorder=3)
ax2.plot(market_index.columns,market_index.iloc[0,:],color='red',linewidth=0.8,label='上证综指',zorder=4)
ax2.plot(market_index.columns,market_index.iloc[2,:]*c/d,color='blue',linewidth=0.8,label='深证成指',zorder=5)
ax2.set_ylabel('指数')
ax2.set_ylim(0,7000)
ax2.legend(loc='upper left')
ax2.set_xlabel('时间')
plt.savefig('个股低成交额所占比例.jpg',dpi=700)

####################################################################






#a = market_index.iloc[0,-1]
#b = market_index.iloc[2,-1]
'''
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid(False)
#ax1.bar(market_volume.index, market_volume.values, width=3,linewidth=0.8,color='yellowgreen',label='全市场成交量',zorder=1)
ax1.plot(stock_PE_median.index, stock_PE_median, color='purple',linewidth=0.8,label='PE中位数',zorder=1)
ax1.set_ylabel('PE中位数')
ax1.set_ylim(0,130)
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.grid(True)
ax2.plot(nationbonds_interest.index,1/nationbonds_interest,color='orange',linewidth=0.8,label='十年国债收益率倒数',zorder=2)
#ax2.plot(market_index.columns,market_index.iloc[0,:],color='purple',linewidth=0.8,label='上证综指',zorder=3)
#ax2.plot(market_index.columns,market_index.iloc[2,:]*a/b,color='purple',linewidth=0.8,label='深证成指',zorder=4)
ax2.set_ylabel('十年国债收益率倒数')
ax2.set_ylim(0,1)
ax2.legend(loc='upper left')
ax2.set_xlabel('时间')
plt.savefig('十年国债收益率倒数.jpg',dpi=700)
'''