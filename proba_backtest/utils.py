import numpy as np
import pandas as pd

def getDailyVol(close, span0=100):
    # daily vol, reindexed to close
    df0=close.index.searchsorted(close.index-pd.Timedelta(days=1))
    df0=df0[df0>0]
    df0=pd.Series(close.index[df0-1], index=close.index[close.shape[0]-df0.shape[0]:])
    df0=close.loc[df0.index]/close.loc[df0.values].values-1 # daily returns
    df0=df0.ewm(span=span0).std()
    return df0

def get_meta_barier(future_window, last_close, min_ret, tp, sl, vertical_zero = True):
    '''
        From https://github.com/Rachnog/Advanced-Deep-Trading/blob/master/bars-labels-diff/Labeling.ipynb
    '''
    if vertical_zero:
        min_ret_situation = [0, 0, 0]
    else:
        min_ret_situation = [0, 0]
        
        
    differences = np.array([(fc - last_close) / last_close for fc in future_window])
    
    # Are there gonna be fluctuations within min_ret???
    min_ret_ups = np.where((differences >= min_ret) == True)[0]
    min_ret_downs = np.where((differences < -min_ret) == True)[0]
  
    if (len(min_ret_ups) == 0) and (len(min_ret_downs) == 0):
        if vertical_zero:
            min_ret_situation[2] = 1
        else:
            if differences[-1] > 0:
                min_ret_situation[0] = 1
            else:
                min_ret_situation[1] = 1            
    else:
        if len(min_ret_ups) == 0: min_ret_ups = [np.inf]
        if len(min_ret_downs) == 0: min_ret_downs = [np.inf]

        if min_ret_ups[0] < min_ret_downs[0]:
            min_ret_situation[0] = 1
        else:
            min_ret_situation[1] = 1
        
    #  Take profit and stop losses indices
    take_profit = np.where((differences >= tp) == True)[0]
    stop_loss = np.where((differences < sl) == True)[0]
    
    # Fluctuation directions coincide with take profit / stop loss actions?
    if min_ret_situation[0] == 1 and len(take_profit) != 0:
        take_action = 1
    elif min_ret_situation[1] == 1 and len(stop_loss) != 0:
        take_action = 1
    else:
        take_action = 0.
    
    return min_ret_situation, take_action, [take_profit, stop_loss]