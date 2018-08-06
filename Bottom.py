
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 11:09:25 2018

@author: Zhehao Li
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import time
from sklearn import linear_model
from pandas.tseries.offsets import Day, MonthEnd
from matplotlib.font_manager import _rebuild
_rebuild()
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 

from WindPy import *
w.start()
#获取上证综指和深证成指
index = w.wsd("000001.SH,399001.SZ", "close", "1990-01-01", "2018-08-06", "")
#获取十年期国债收益率
nation_bonds = w.edb("S0059749", "1990-01-01", "2018-08-06","Fill=Previous")
#获取M2
M2_data = w.edb("M0001384", "1990-01-01", "2018-08-06","Fill=Previous")
#人民币兑美元汇率，在岸，离岸
CNY = w.wsd("USDCNY.IB,USDCNH.FX", "close", "1990-01-01", "2018-08-06", "")
w.close()

market_index = pd.DataFrame(data=index.Data, columns=index.Times, index=index.Codes)
nationbonds_interest = pd.DataFrame(data=nation_bonds.Data, columns=nation_bonds.Times,index=nation_bonds.Codes)
m2_data = pd.DataFrame(data=M2_data.Data, columns=M2_data.Times, index=M2_data.Codes)
cny_data = pd.DataFrame(data=CNY.Data, columns=CNY.Times, index=CNY.Codes)

market_index.columns = pd.DatetimeIndex(market_index.columns)
nationbonds_interest.columns = pd.DatetimeIndex(nationbonds_interest.columns)
m2_data.columns = pd.DatetimeIndex(m2_data.columns)
cny_data.columns = pd.DatetimeIndex(cny_data.columns)

import cx_Oracle
dsn = cx_Oracle.makedsn('10.88.102.160','1521','gfdwdb1')
conn = cx_Oracle.connect('jcquery','83936666',dsn)

sql_price = "select s_info_windcode, trade_dt, s_dq_close, s_dq_amount, s_dq_volume from gfwind.AShareEodPrices"
sql_accounting = "select s_info_windcode, trade_dt, s_val_mv, s_val_pe_ttm, s_val_pb_new from gfwind.AShareEODDerivativeIndicator" 
sql_ipo_date = "select s_info_windcode, s_info_listdate from gfwind.AShareDescription"
sql_ipo_price = "select s_info_windcode, s_ipo_price from gfwind.AShareIPO"

stock_price = pd.read_sql(sql_price,conn)
stock_accounting = pd.read_sql(sql_accounting,conn)
ipo_date = pd.read_sql(sql_ipo_date,conn)
ipo_price = pd.read_sql(sql_ipo_price,conn)

###############################################################################


###############################################################################
'''
0. 准备工作
'''
# 合并数据
#股票日收盘价，成交量，成交额，PE，PB，市值
finance_data = pd.merge(stock_price,stock_accounting,on=['S_INFO_WINDCODE','TRADE_DT'])
#股票上市日期和发行价
ipo_data = pd.merge(ipo_date,ipo_price,on='S_INFO_WINDCODE')


# finance_data 所有数据

#提取日收盘价
stock_close_data = pd.pivot_table(finance_data,values='S_DQ_CLOSE',index='S_INFO_WINDCODE',columns='TRADE_DT')                   
stock_close_data = stock_close_data.sort_index(axis=1)
stock_close_data.columns = pd.DatetimeIndex(stock_close_data.columns)
stock_close_number = stock_close_data.count(axis=0)


#绘图函数

def Plot_Stock(data_a,data_b,label_a,label_name):
    # c,d为调整系数，是上证综指和深证成指头部对齐
    c = data_b.iloc[0,-1]
    d = data_b.iloc[1,-1]
    
    fig = plt.figure(figsize=(20,5))
    ax1 = fig.add_subplot(111)
    ax1.grid(False)
    ax1.bar(data_a.index,data_a,width=3.5,linewidth=3,color='yellowgreen',label=label_a,zorder=1)
    ax1.set_ylabel(label_name)
    ax1.set_ylim(0,1.1*data_a.max())
    ax1.legend(loc='upper right')
    
    ax2 = ax1.twinx()
    ax2.grid(True)
    ax2.plot(data_b.columns,data_b.iloc[0,:],color='red',linewidth=0.8,label='上证综指',zorder=5)
    ax2.plot(data_b.columns,data_b.iloc[1,:]*c/d,color='blue',linewidth=0.8,label='深证成指',zorder=6)
    ax2.set_ylabel('指数')
    ax2.set_ylim(0,7000)
    ax2.legend(loc='upper left')
    ax2.set_xlabel('时间')
    plt.savefig(label_name+'.jpg',dpi=1000)


###############################################################################
'''
1.低价股比例
'''
#筛选出低价股
low_price=2 #低价股标准
low_price_data = stock_close_data[stock_close_data<low_price]
#低价股数量
low_price_number = low_price_data.count(axis=0)
low_price_percent = low_price_number/stock_close_number
low_price_percent.index = pd.DatetimeIndex(low_price_percent.index) # pd.Series


#绘图
Plot_Stock(data_a=low_price_percent,data_b=market_index,label_a='低价股比例',label_name='全市场低价股占比')


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

#绘图
Plot_Stock(data_a=low_PB_percent,data_b=market_index,label_a='破净股比例',label_name='全市场破净股比例')

###############################################################################
'''
3.M2/总市值中位数
'''

#提取市值数据
market_value_data = pd.pivot_table(finance_data,values='S_VAL_MV',index='S_INFO_WINDCODE',columns='TRADE_DT')
market_value_data = market_value_data.sort_index(axis=1)
market_value_median = market_value_data.median(axis=0)

#将市值数据调整为月度数据
market_value_median = market_value_median.reset_index()
market_value_median['MONTH'] = market_value_median['TRADE_DT'].apply(lambda x:x[:6])
market_value_median = pd.merge(market_value_median.groupby(['MONTH'])['TRADE_DT'].max().reset_index(),market_value_median,on=['TRADE_DT'],how='left')
market_value_median.drop(['MONTH_x','MONTH_y'],axis=1,inplace=True)
market_value_median.columns = ['TRADE_DT','PE_MEDIAN']
market_value_median = market_value_median.set_index('TRADE_DT')
market_value_median.index = pd.DatetimeIndex(market_value_median.index)

m2_data = m2_data.T
#从1996年开始计算
m2_data[6:]
# M2数据一般比市值数据滞后两个月
market_value_median[61:-2]

assert len(m2_data[6:]) == len(market_value_median[61:-2])
M2_MV_ratio = m2_data.values[6:]/market_value_median.values[61:-2] #numpy.array

#绘图
fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(111)
ax1.grid(False)
ax1.plot(market_value_median.index[61:-2],market_value_median[61:-2], color='purple',linewidth=0.8,label='总市值中位数',zorder=3)
ax1.set_ylabel('总市值中位数')
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.grid(True)
ax2.plot(market_value_median.index[61:-2],M2_MV_ratio,color='orange',linewidth=0.8,label='M2/总市值中位数',zorder=4)
ax2.set_ylabel('M2/总市值中位数')
ax2.legend(loc='upper left')
ax2.set_xlabel('时间')
plt.savefig('M2_总市值中位数.jpg',dpi=1000)


###############################################################################
'''
4.PE中位数&十年国债收益率倒数
'''

#提取PE
stock_PE_data = pd.pivot_table(finance_data,values='S_VAL_PE_TTM',index='S_INFO_WINDCODE',columns='TRADE_DT')
stock_PE_data = stock_PE_data.sort_index(axis=1)
stock_PE_data.columns = pd.DatetimeIndex(stock_PE_data.columns)
stock_PE_median = stock_PE_data.median(axis=0)

nationbonds_interest = nationbonds_interest.T

fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(111)
ax1.grid(False)
ax1.plot(stock_PE_median.index,stock_PE_median, color='purple',linewidth=0.8,label='市场PE中位数',zorder=3)
ax1.set_ylabel('市场PE中位数')
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.grid(True)
ax2.plot(nationbonds_interest.index,nationbonds_interest,color='orange',linewidth=0.8,label='十年期国债收益率',zorder=4)
ax2.set_ylabel('十年期国债收益')
ax2.legend(loc='upper left')
ax2.set_xlabel('时间')
plt.savefig('PE中位数和十年国债收益率的比较.jpg',dpi=1000)

###############################################################################

def Plot_Volume(data_a,data_b,index_data,label_a,label_b,label_name):
    a = data_a.values[0]
    b = data_b.values[0]
    
    c = market_index.iloc[0,-1]
    d = market_index.iloc[1,-1]
    
    fig = plt.figure(figsize=(20,5))
    ax1 = fig.add_subplot(111)
    ax1.grid(False)
    ax1.bar(data_a.index,data_a*1.3*b/a,width=3.5,linewidth=0.8,color='yellowgreen',label=label_a,zorder=1)
    ax1.bar(data_b.index,data_b,width=3.5,linewidth=0.8,color='orange',label=label_b,zorder=2)
    ax1.set_ylabel(label_name)
    ax1.legend(loc='upper right')
    
    ax2 = ax1.twinx()
    ax2.grid(True)
    ax2.plot(index_data.columns,index_data.iloc[0,:],color='red',linewidth=0.8,label='上证综指',zorder=5)
    ax2.plot(index_data.columns,index_data.iloc[1,:]*c/d,color='blue',linewidth=0.8,label='深证成指',zorder=6)
    ax2.set_ylabel('指数')
    ax2.set_ylim(0,7000)
    ax2.legend(loc='upper left')
    ax2.set_xlabel('时间')
    plt.savefig(label_name+'.jpg',dpi=1000)

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

#绘图
Plot_Volume(data_a=total_amount,data_b=total_volume,index_data=market_index,label_a='市场总成交额',label_b='市场总成交量',label_name='市场成交量&成交额')


###############################################################################
'''
6.个股低成交额&成交量占比（个股流动性）
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

#绘图
Plot_Volume(data_a=low_amount_percent,data_b=low_volume_percent,index_data=market_index,label_a='低成交额个股占比',label_b='低成交量个股占比',label_name='低成交量&成交额股票占比')


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
decline_median.index = pd.to_datetime(decline_median.index,format="%Y%m")


#绘图
c = market_index.iloc[0,-1]
d = market_index.iloc[1,-1]

fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(111)
ax1.grid(False)
ax1.bar(decline_median.index,decline_median,width=14,linewidth=10,color='yellowgreen',label='个股最大跌幅中位数',zorder=1)
ax1.set_ylabel('个股最大跌幅中位数')
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.grid(True)
ax2.plot(market_index.columns,market_index.iloc[0,:],color='red',linewidth=0.8,label='上证综指',zorder=5)
ax2.plot(market_index.columns,market_index.iloc[1,:]*c/d,color='blue',linewidth=0.8,label='深证成指',zorder=6)
ax2.set_ylabel('指数')
ax2.set_ylim(0,7000)
ax2.legend(loc='upper left')
ax2.set_xlabel('时间')
plt.savefig('个股单月最大跌幅中位数.jpg',dpi=1000)



###############################################################################
'''
8.次新股(上市一年内)PE（市场人气）
'''

ipo_data = ipo_data.sort_values('S_INFO_WINDCODE').reset_index()
ipo_data = ipo_data.drop('index',axis=1)
ipo_data['S_INFO_LISTDATE'] = pd.DatetimeIndex(ipo_data['S_INFO_LISTDATE'])

#筛选出时间横截面的次新股
subnew_stock=pd.DataFrame()
for day in stock_close_data.columns:
    period = day - ipo_data['S_INFO_LISTDATE']
    period = period.apply(lambda x:str(x)[:-14])
    period = pd.to_numeric(period)
    subnew_stock[day] = ipo_data[period<365]['S_INFO_WINDCODE']
    subnew_stock[day] = subnew_stock[day][period>1]

subnew_number = subnew_stock.count(axis=0)

#计算次新股PE中位数
subnew_PE_median = []
for day in stock_PE_data.columns:
    temp_PE = stock_PE_data[day].reset_index()
    temp_PE.columns=['S_INFO_WINDCODE','PE']
    temp_subnew = pd.DataFrame(subnew_stock[day].dropna())
    temp_subnew.columns = ['S_INFO_WINDCODE']
    pe = pd.merge(temp_subnew,temp_PE,how='left',on='S_INFO_WINDCODE')
    subnew_PE_median.append(pe.iloc[:,-1].median(axis=0))

#次新股PE中位数    
subnew_PE_median = np.array(subnew_PE_median)
subnew_PE_median = pd.Series(subnew_PE_median,index=stock_PE_median.index)

#次新股PE溢价
differences_PE_median = subnew_PE_median-stock_PE_median.values


fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(111)
ax1.grid(False)
ax1.bar(subnew_PE_median.index[600:],differences_PE_median[600:],width=14,linewidth=10,color='yellowgreen',label='次新股PE中位数溢价',zorder=1)
ax1.set_ylabel('次新股PE中位数溢价')
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.grid(True)
ax2.plot(subnew_PE_median.index[600:],subnew_PE_median[600:],color='purple',linewidth=0.8,label='次新股PE中位数',zorder=5)
ax2.plot(stock_PE_median.index,stock_PE_median,color='orange',linewidth=0.8,label='市场PE中位数',zorder=6)
ax2.set_ylabel('PE中位数')
ax2.legend(loc='upper left')
ax2.set_xlabel('时间')
plt.savefig('次新股PE中位数.jpg',dpi=1000)



###############################################################################
'''
9. 次新股破发率（市场人气）
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

    lower_stock = temp_data[temp_data['S_IPO_PRICE']>temp_data['PRICE']]['S_INFO_WINDCODE']
    return len(lower_stock)


lower_ipo_stock = []
for day in stock_close_data.columns:
    lower_ipo_stock.append(Lower_Stock(ipo_data,subnew_stock,stock_close_data,day))

#次新股破发比例
lower_ipo_percent = np.array(lower_ipo_stock).reshape(-1)/subnew_number.values.reshape(-1)
lower_ipo_percent = pd.Series(lower_ipo_percent.reshape(-1),index=stock_close_data.columns)


c = market_index.iloc[0,-1]
d = market_index.iloc[1,-1]


fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(111)
ax1.grid(False)
ax1.bar(lower_ipo_percent.index[-1095:],lower_ipo_percent[-1095:],width=3,linewidth=0.8,color='yellowgreen',label='次新股破发率',zorder=2)
ax1.set_ylabel('次新股破发率')
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.grid(True)
ax2.plot(market_index.columns[-1095:],market_index.iloc[0,-1095:],color='red',linewidth=0.8,label='上证综指',zorder=5)
ax2.plot(market_index.columns[-1095:],market_index.iloc[1,-1095:]*c/d,color='blue',linewidth=0.8,label='深证成指',zorder=6)
ax2.set_ylabel('指数')
ax2.set_ylim(0,7000)
ax2.legend(loc='upper left')
ax2.set_xlabel('时间')
plt.savefig('次新股破发率.jpg',dpi=1000)

####################################################################
'''
10. 汇率和指数
'''

c = market_index.iloc[0,-1]
d = market_index.iloc[1,-1]


fig = plt.figure(figsize=(20,5))
ax1 = fig.add_subplot(111)
ax1.grid(True)
ax1.plot(market_index.columns[-1095:],market_index.iloc[0,-1095:],color='yellowgreen',linewidth=0.8,label='上证综指',zorder=1)
ax1.plot(market_index.columns[-1095:],market_index.iloc[1,-1095:]*c/d,color='orange',linewidth=0.8,label='深证成指',zorder=2)
ax1.set_ylabel('指数')
ax1.set_ylim(0,7000)
ax1.legend(loc='upper right')

ax2 = ax1.twinx()
ax2.grid(False)
ax2.plot(cny_data.columns[-1095:],cny_data.iloc[0,-1095:],color='red',linewidth=0.8,label='美元兑在岸人民币',zorder=3)
ax2.plot(cny_data.columns[-1095:],cny_data.iloc[1,-1095:],color='blue',linewidth=0.8,label='美元兑离岸人民币',zorder=3)
ax2.set_ylabel('汇率变动')
ax2.set_ylim(4,10)
ax2.legend(loc='upper left')
ax2.set_xlabel('时间')
plt.savefig('汇率与指数.jpg',dpi=1000)

