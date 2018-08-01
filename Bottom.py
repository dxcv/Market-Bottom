
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
from pandas.tseries.offsets import Day, MonthEnd

from matplotlib.font_manager import _rebuild
_rebuild()
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 
'''
from WindPy import * 
w.start()
nationbonds = w.edb("M0325687", "2007-01-04", "2018-07-30")
w.close()
'''

# 导入数据
price_data = pd.read_pickle('/Users/trevor/Downloads/price.pkl')
indicator_data = pd.read_pickle('/Users/trevor/Downloads/indicator.pkl')
market_index = pd.read_excel('/Users/trevor/Downloads/市场底部特征研究.xlsx',sheet_name='沪深300')
M2_data = pd.read_excel('/Users/trevor/Downloads/M2指标.xlsx')
IPO_data = pd.read_excel('/Users/trevor/Downloads/A股上市时间.xlsx')
amount_data = pd.read_csv('/Users/trevor/Downloads/eodprices.csv')

###############################################################################
'''
0. 准备工作
'''
# 合并数据
finance_data = pd.merge(price_data,indicator_data,on=['S_INFO_WINDCODE','TRADE_DT'])
amount_data['TRADE_DT'] = amount_data['TRADE_DT'].apply(lambda x:str(x))
finance_data = pd.merge(finance_data,amount_data,how='left',on=['S_INFO_WINDCODE','TRADE_DT'])

#提取日收盘价
stock_close_data = pd.pivot_table(finance_data,values='S_DQ_CLOSE',index='S_INFO_WINDCODE',columns='TRADE_DT')                   
stock_close_data = stock_close_data.sort_index(axis=1)
stock_close_data.columns = pd.DatetimeIndex(stock_close_data.columns)
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
stock_PE_data.columns = pd.DatetimeIndex(stock_PE_data.columns)
stock_PE_median = stock_PE_data.median(axis=0)
stock_PE_median.index = pd.DatetimeIndex(stock_PE_median.index)
stock_PE_average = stock_PE_data.mean(axis=0)
stock_PE_average.index = pd.DatetimeIndex(stock_PE_average.index)

#十年国债收益率
nationbonds_interest = pd.Series(np.array(nationbonds.Data).reshape(-1),index=nationbonds.Times)
nationbonds_interest.index = pd.DatetimeIndex(nationbonds_interest.index)

###############################################################################
'''
5.全市场成交额&成交量（市场人气）
'''
#所有股票成交额&成交量
stock_volume_data = pd.pivot_table(finance_data,values='S_DQ_VOLUME',index='S_INFO_WINDCODE',columns='TRADE_DT')
stock_volume_data = stock_volume_data.sort_index(axis=1)

stock_amount_data = pd.pivot_table(finance_data,values='S_DQ_AMOUNT',index='S_INFO_WINDCODE',columns='TRADE_DT')
stock_amount_data = stock_amount_data.sort_index(axis=1)
#市场总成交额&成交量
total_volume = stock_volume_data.sum(axis=0)
total_volume.index = pd.DatetimeIndex(total_volume.index)

total_amount = stock_amount_data.sum(axis=0)
total_amount.index = pd.DatetimeIndex(total_amount.index)

###############################################################################
'''
6.个股成交额&成交量（个股流动性）
'''
#筛选低成交量&成交额个股
low_volume = 1000 #(手)
low_volume_stock = stock_volume_data[stock_volume_data<low_volume]
low_volume_number = low_volume_stock.count(axis=0)
low_volume_percent = low_volume_number/stock_close_number
low_volume_percent.index = pd.DatetimeIndex(low_volume_percent.index)

low_amount = 1000 #(千元)
low_amount_stock = stock_amount_data[stock_amount_data<low_amount]
low_amount_number = low_amount_stock.count(axis=0)
low_amount_percent = low_amount_number/stock_close_number
low_amount_percent.index = pd.DatetimeIndex(low_amount_percent.index)

###############################################################################
'''
7.个股区间最大跌幅中位数（月区间）
'''
temp_finance_data = finance_data
temp_finance_data['MONTH'] = temp_finance_data['TRADE_DT'].apply(lambda x:x[:6])

price_max = temp_finance_data.groupby(['MONTH','S_INFO_WINDCODE'])['S_DQ_CLOSE'].max()
price_min = temp_finance_data.groupby(['MONTH','S_INFO_WINDCODE'])['S_DQ_CLOSE'].min()
price_max = pd.pivot_table(price_max.reset_index(),values='S_DQ_CLOSE',index='S_INFO_WINDCODE',columns='MONTH')
price_min = pd.pivot_table(price_min.reset_index(),values='S_DQ_CLOSE',index='S_INFO_WINDCODE',columns='MONTH')

temp_finance_data = temp_finance_data.set_index('TRADE_DT')

index_max = temp_finance_data.groupby(['MONTH','S_INFO_WINDCODE'])['S_DQ_CLOSE'].idxmax()
index_min = temp_finance_data.groupby(['MONTH','S_INFO_WINDCODE'])['S_DQ_CLOSE'].idxmin()
index_max = index_max.reset_index()
index_min = index_min.reset_index()
index_max.columns = ['MONTH','S_INFO_WINDCODE','DATE']
index_min.columns = ['MONTH','S_INFO_WINDCODE','DATE']
index_max['DATE'] = index_max['DATE'].apply(lambda x:int(x))
index_min['DATE'] = index_min['DATE'].apply(lambda x:int(x))
index_max = pd.pivot_table(index_max,values='DATE',index='S_INFO_WINDCODE',columns='MONTH')
index_min = pd.pivot_table(index_min,values='DATE',index='S_INFO_WINDCODE',columns='MONTH')

#计算个股月区间最大跌幅中位数
high = price_max[index_max<index_min]
low = price_min[index_max<index_min]
decline_percent = (high - low)/high
decline_median = decline_percent.median(axis=0)
decline_median.index = pd.DatetimeIndex(start='2006-12-30',end='2018-07-01',freq='M')


###############################################################################
'''
8.次新股(上市四个月内)PE（市场人气）
'''
#获取所有股票发行价格和发行时间
IPO_data['IPO_DATE'] = pd.DatetimeIndex(IPO_data['IPO_DATE'])

#筛选出时间横截面的次新股
subnew_stock=pd.DataFrame()
for day in stock_close_data.columns:
    period = day - IPO_data['IPO_DATE']
    subnew_stock[day] = IPO_data[period<365*Day()]['S_INFO_WINDCODE']
    subnew_stock[day] = subnew_stock[day][period>1*Day()]

subnew_number = subnew_stock.count(axis=0)


subnew_PE_median = []
subnew_PE_average = []
for day in stock_close_data.columns:
    temp_PE = stock_PE_data[day].reset_index()
    temp_PE.columns=['S_INFO_WINDCODE','PE']
    temp_subnew = pd.DataFrame(subnew_stock[day].dropna())
    temp_subnew.columns = ['S_INFO_WINDCODE']
    pe = pd.merge(temp_subnew,temp_PE,how='left',on='S_INFO_WINDCODE')
    subnew_PE_median.append(pe.iloc[:,-1].median(axis=0))
    subnew_PE_average.append(pe.iloc[:,-1].mean(axis=0))
    
subnew_PE_median = np.array(subnew_PE_median)
subnew_PE_average = np.array(subnew_PE_average)

differences_PE_median = subnew_PE_median-stock_PE_median.values
differences_PE_average = subnew_PE_average-stock_PE_average.values
###############################################################################
'''
9. 股票破发率（市场人气）
'''
#时间横截面破发股比例
def Lower_Stock(IPO_data,subnew_stock,stock_close_data,day):
    
    temp_subnew = subnew_stock[day].dropna() 
    temp_subnew = temp_subnew.reset_index()
    temp_subnew.columns = ['index','S_INFO_WINDCODE']
    temp_subnew = temp_subnew.drop('index',axis=1)
    
    temp_stock = stock_close_data[day]
    temp_stock = temp_stock.reset_index()
    temp_stock.columns = ['S_INFO_WINDCODE','PRICE']
    
    temporary = pd.merge(temp_subnew,temp_stock,how='left',on='S_INFO_WINDCODE')
    temp_IPO = IPO_data
    temp_data = pd.merge(temporary,temp_IPO,how='left',on='S_INFO_WINDCODE')

    lower_stock = temp_data[temp_data['IPO_PICE']>temp_data['PRICE']]['S_INFO_WINDCODE']
    return len(lower_stock)


lower_ipo_stock = []
for day in stock_close_data.columns:
    lower_ipo_stock.append(Lower_Stock(IPO_data,subnew_stock,stock_close_data,day))


lower_ipo_percent = np.array(lower_ipo_stock)/subnew_number.values*100


###############################################################################

c = market_index.iloc[0,-1]
d = market_index.iloc[2,-1]

#assert stock_close_data.columns.shape==lower_ipo_percent.shape
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid(False)
ax1.bar(decline_median.index,decline_median,width=10,linewidth=10,color='yellowgreen',label='个股单月最大跌幅中位数',zorder=1)
#ax1.bar(low_volume_percent.index,low_volume_percent/1.5,width=3.5,linewidth=0.8,color='orange',label='低成交量股票占比',zorder=2)
#ax1.plot(subnew_PE_median.index,subnew_PE_median, color='purple',linewidth=0.8,label='次新股PE中位数',zorder=3)
ax1.set_ylabel('个股单月最大跌幅中位数')#低成交额占比(小于100W)
#ax1.set_ylim(0,130)
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.grid(True)
#ax2.plot(stock_PE_average.index,subnew_PE_average, color='purple',linewidth=0.8,label='次新股PE平均值',zorder=3)
#ax2.plot(stock_PE_average.index,stock_PE_average,color='orange',linewidth=0.8,label='市场PE平均值',zorder=4)
ax2.plot(market_index.columns,market_index.iloc[0,:],color='red',linewidth=0.8,label='上证综指',zorder=5)
ax2.plot(market_index.columns,market_index.iloc[2,:]*c/d,color='blue',linewidth=0.8,label='深证成指',zorder=6)
ax2.set_ylabel('指数')
#ax2.set_ylim(0,7000)
ax2.legend(loc='upper left')
ax2.set_xlabel('时间')
plt.savefig('个股单月最大跌幅中位数.jpg',dpi=700)

####################################################################



