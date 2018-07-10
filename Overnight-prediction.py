from quantopian.algorithm import attach_pipeline, pipeline_output  
from quantopian.pipeline import Pipeline  
import numpy as np  
import quantopian.optimize as opt
import pandas as pd  
from quantopian.pipeline.filters import Q1500US
from quantopian.pipeline.data import Fundamentals
from quantopian.pipeline.classifiers.fundamentals import Sector
from quantopian.pipeline.data import morningstar as mstar
from quantopian.pipeline.factors import AverageDollarVolume

num_short = 75
num_long = 75
num_tradable = num_short + num_long

# compute weights, helper function  
def to_weights_closing(factor):  
    demeaned_vals = factor - factor.mean()  
    return demeaned_vals / demeaned_vals.abs().sum()
        

def initialize(context): 
    #set a realistic commission and slippage
    set_commission(commission.PerShare(cost=0.0035, min_trade_cost=0.35))
    set_slippage(slippage.VolumeShareSlippage(volume_limit=0.025, price_impact=0.1))
    #schedule function so that I open positions before market closes  
    schedule_function(market_close_trades, date_rules.every_day(), time_rules.market_close(minutes=1)) 
    #schedule function to change position  
    #when the market opens the next morning  
    schedule_function(market_open_trades, date_rules.every_day(), time_rules.market_open(minutes=1))
    #schedule function to close all positions 30 minutes  
    #after the market opens everyday 
    schedule_function(close_all, date_rules.every_day(), time_rules.market_open(minutes=31))
    #leverage for long positions  
    context.long_leverage = 0.75
    #leverage for short positions  
    context.short_leverage = -0.75
    #create pipeline  
    my_pipe = make_pipeline()  
    attach_pipeline(my_pipe, 'stocks_1500') 
    
def make_pipeline():
    # Base universe set to the Q1500US.
    dollar_volume = AverageDollarVolume(window_length=5)
    is_liquid = dollar_volume.top(150)
    base_universe = Q1500US()
    sector = mstar.asset_classification.morningstar_sector_code.latest
    technology = sector.element_of([301])
    industrials = sector.element_of([310])
    utilities = sector.element_of([207])
    energy = sector.element_of([309])
    healthcare = sector.element_of([206])
    financials = sector.element_of([103])
    
    # Screen only securities tradable for the day
    securities_to_trade = (base_universe & is_liquid & Sector().element_of( [103, 206, 207, 301, 309, 310]))
    
    pipe = Pipeline(  
        columns={
            'Technology': technology,
            'Industrials': industrials,
            'Utilities': utilities,
            'Energy': energy,
            'Healthcare': healthcare,
            'Financials': financials,
        },  
        screen=securities_to_trade,  
    )
    return pipe

def before_trading_start(context, data):  
    # Call pipelive_output to get the output  
    context.output = pipeline_output('stocks_1500')
    context.base_universe = context.output.index.tolist()
    context.max_long_sec = 75
    context.max_short_sec = 75

def market_close_trades(context, data): 
    '''
    This function rebalances my portfolio when the market closes  
    I go long for the best-performing 5% stocks 
    if the "last 30 minutes of trading returns" are positive.
    I go short for the worst-performing 5% stocks when  
    the "last 30 minutes of trading returns" are negative.
    '''
    # get close prices for last 30 minutes
    returns = data.history(context.base_universe, 'close', 60, '1m').pct_change()
    # Only want the current 30 minute return so use the loc method to select a single row
    # Additionally, this transposes the securities so they become the index of the returned series.
    now = get_datetime()
    last_30_min_returns = returns.loc[now]
    # Select securities where we have a gain in the last 30 minutes of trading  
    increasing_last_30_mins = last_30_min_returns.loc[last_30_min_returns > 0]
    # Select securities where we have a loss in the last 30 minutes of trading  
    decreasing_last_30_mins = last_30_min_returns.loc[last_30_min_returns < 0]
    overnight_returns = increasing_last_30_mins.append(decreasing_last_30_mins)
    # factor to weights so that they are variable
    # if a stock gives me a particular return, I want to
    # invest more in it
    weights = to_weights_closing(overnight_returns)  
    longs  = weights[ weights > 0 ]  
    shorts = weights[ weights < 0 ].abs()

    # limit number of securities  
    if context.max_long_sec:  
        longs  = longs.sort_values(ascending=False).head(context.max_long_sec)  
    if context.max_short_sec:  
        shorts = shorts.sort_values(ascending=False).head(context.max_short_sec)

    # normalize weights to 1.  
    longs  /= longs.sum()  
    shorts /= shorts.sum()  
    longs  /= 2  
    shorts /= 2
    # If last "30 minutes of trading" returns are positive go long  
    # based on my long list
    for stock in longs.index:  
        order_target_percent(stock, longs[stock])  
    for stock in shorts.index:  
        order_target_percent(stock, shorts[stock])
        
def market_open_trades(context, data): 
    '''
    This function rebalances my portfolio when the market opens  
    I go short if the "overnight returns" are positive and  
    I go long otherwise.  
    This happens only for the best-performing 5% stocks and  
    for the worst-performing 5% stocks 
    '''
    # get close prices for the open today and the close of yesterday
    # NOTE this assumes this is run at 1 minute after open
    close_prices = data.history(context.base_universe, 'close', 2, '1m')
    # Find the percent return using the dataframe method 'pct_change'
    returns = close_prices.pct_change()
    # Only want the last (current) return so use the iloc method to select a single row
    # Additionally, this transposes the securities so they become the index of the returned series.
    overnight_returns = returns.iloc[-1]
    positive_overnight_returns = overnight_returns[overnight_returns > 0]
    negative_overnight_returns = overnight_returns[overnight_returns < 0]
    # Select securities to long and short. Use index.tolist to just get a list of securities
    longs = negative_overnight_returns.nsmallest(num_long).index.tolist() 
    shorts = positive_overnight_returns.nlargest(num_short).index.tolist()
    # Compute weights  
    long_weight = context.long_leverage / num_long
    short_weight = context.short_leverage / num_short
    # If overnight returns are positive go long  
    # based on my long list
    for stock in longs:  
        order_target_percent(stock, long_weight)  
    # If overnight returns are negative go short  
    # based on my short list 
    for stock in shorts:  
        order_target_percent(stock, short_weight) 
def close_all(context, data): 
    '''
    This function rebalances my portfolio by closing all positions  
    1 minute after the market opens in the next morning
    '''
    # Close every position 30 minutes after the market opens  
    for stock in context.portfolio.positions:  
            order_target_percent(stock, 0)
