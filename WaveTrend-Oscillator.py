import talib
'''
Implementation of the WaveTrend Oscillator strategy: https://www.tradingview.com/script/2KE8wTuF-Indicator-WaveTrend-Oscillator-WT/
'''
# ---------------------------------------------------
n1, n2, period, stock = 10, 21, 60, sid(24)
# ---------------------------------------------------
def initialize(context):
    schedule_function(trade, date_rules.week_start(), time_rules.market_open())

def trade(context, data):
    ob = 57 #"Over Bought Level"  
    os = -57 #"Over Sold Level"
    if get_open_orders(): return
    close = data.history(stock, 'close', period + 1, '1d').dropna()
    low = data.history(stock, 'low', period + 1, '1d').dropna()
    high = data.history(stock, 'high', period + 1, '1d').dropna()
    price = data.history(stock, 'price', period + 1, '1d').dropna()
    ap = (high + low + close) / 3
    esa = talib.EMA(ap, timeperiod=n1)
    d = talib.EMA(abs(ap - esa), timeperiod=n1)
    ci = (ap - esa) / (0.015 * d)
    wt1 = talib.EMA(ci, timeperiod=n2)
    record(wt1 = wt1[-1], ob = ob,os = os)
    if data.can_trade(stock):
        if  wt1[-1] > os:
            order_target_percent(stock, 2)
        elif wt1[-1] < ob:
            order_target_percent(stock, -1)
