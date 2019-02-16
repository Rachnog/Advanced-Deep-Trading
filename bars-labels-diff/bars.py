import pandas as pd
import numpy as np
import datetime

class BarSeries(object):
    '''
        Base class for resampling ticks dataframe into OHLC(V)
        using different schemes. This particular class implements
        standard time bars scheme.
        See: https://www.wiley.com/en-it/Advances+in+Financial+Machine+Learning-p-9781119482086
    '''
    
    def __init__(self, df, datetimecolumn = 'DateTime'):
        self.df = df
        self.datetimecolumn = datetimecolumn
    
    def process_column(self, column_name, frequency):
        return self.df[column_name].resample(frequency, label='right').ohlc()
    
    def process_volume(self, column_name, frequency):
        return self.df[column_name].resample(frequency, label='right').sum()
    
    def process_ticks(self, price_column = 'Price', volume_column = 'Size', frequency = '15Min'):
        price_df = self.process_column(price_column, frequency)
        volume_df = self.process_volume(volume_column, frequency)
        price_df['volume'] = volume_df
        return price_df
    

class TickBarSeries(BarSeries):
    '''
        Class for generating tick bars based on bid-ask-size dataframe
    '''
    def __init__(self, df, datetimecolumn = 'DateTime', volume_column = 'Size'):
        self.volume_column = volume_column
        super(TickBarSeries, self).__init__(df, datetimecolumn)
    
    def process_column(self, column_name, frequency):
        res = []
        for i in range(frequency, len(self.df), frequency):
            sample = self.df.iloc[i-frequency:i]
            v = sample[self.volume_column].values.sum()
            o = sample[column_name].values.tolist()[0]
            h = sample[column_name].values.max()
            l = sample[column_name].values.min()
            c = sample[column_name].values.tolist()[-1]
            d = sample.index.values[-1]
            
            res.append({
                self.datetimecolumn: d,
                'open': o,
                'high': h,
                'low': l,
                'close': c,
                'volume': v
            })

        res = pd.DataFrame(res).set_index(self.datetimecolumn)
        return res

    
    def process_ticks(self, price_column = 'Price', volume_column = 'Size', frequency = '15Min'):
        price_df = self.process_column(price_column, frequency)
        return price_df    
    

class VolumeBarSeries(BarSeries):
    '''
        Class for generating volume bars based on bid-ask-size dataframe
    '''
    def __init__(self, df, datetimecolumn = 'DateTime', volume_column = 'Size'):
        self.volume_column = volume_column
        super(VolumeBarSeries, self).__init__(df, datetimecolumn)
               
    def process_column(self, column_name, frequency):
        res = []
        buf = []
        start_index = 0.
        volume_buf = 0.
        for i in range(len(self.df[column_name])):

            pi = self.df[column_name].iloc[i]
            vi = self.df[self.volume_column].iloc[i]
            di = self.df.index.values[i]
            
            buf.append(pi)
            volume_buf += vi
            
            if volume_buf >= frequency:
                
                o = buf[0]
                h = np.max(buf)
                l = np.min(buf)
                c = buf[-1]
                
                res.append({
                    self.datetimecolumn: di,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,  
                    'volume': volume_buf
                })
                
                buf, volume_buf = [], 0.

        res = pd.DataFrame(res).set_index(self.datetimecolumn)
        return res
    
    def process_ticks(self, price_column = 'Price', volume_column = 'Size', frequency = '15Min'):
        price_df = self.process_column(price_column, frequency)
        return price_df    
    
    

class DollarBarSeries(BarSeries):
    '''
        Class for generating "dollar" bars based on bid-ask-size dataframe
    '''
    def __init__(self, df, datetimecolumn = 'DateTime', volume_column = 'Size'):
        self.volume_column = volume_column
        super(DollarBarSeries, self).__init__(df, datetimecolumn)
               
    def process_column(self, column_name, frequency):
        res = []
        buf, vbuf = [], []
        start_index = 0.
        dollar_buf = 0.
        for i in range(len(self.df[column_name])):

            pi = self.df[column_name].iloc[i]
            vi = self.df[self.volume_column].iloc[i]
            di = self.df.index.values[i]
            
            dvi = pi * vi
            buf.append(pi)
            vbuf.append(vi)
            dollar_buf += dvi
            
            if dollar_buf >= frequency:
                
                o = buf[0]
                h = np.max(buf)
                l = np.min(buf)
                c = buf[-1]
                v = np.sum(vbuf)
                
                res.append({
                    self.datetimecolumn: di,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,
                    'volume': v,
                    'dollar': dollar_buf
                })
                
                buf, vbuf, dollar_buf = [], [], 0.

        res = pd.DataFrame(res).set_index(self.datetimecolumn)
        return res 
    
    def process_ticks(self, price_column = 'Price', volume_column = 'Size', frequency = '15Min'):
        price_df = self.process_column(price_column, frequency)
        return price_df    
    
    
class ImbalanceTickBarSeries(BarSeries):
    '''
        Class for generating imbalance tick bars based on bid-ask-size dataframe
    '''
    def __init__(self, df, datetimecolumn = 'DateTime', volume_column = 'Size'):
        self.volume_column = volume_column
        super(ImbalanceTickBarSeries, self).__init__(df, datetimecolumn)
        
    def get_bt(self, data):
        s = np.sign(np.diff(data))
        for i in range(1, len(s)):
            if s[i] == 0:
                s[i] = s[i-1]
        return s

    def get_theta_t(self, bt):
        return np.sum(bt)

    def ewma(self, data, window):

        alpha = 2 /(window + 1.0)
        alpha_rev = 1-alpha

        scale = 1/alpha_rev
        n = data.shape[0]

        r = np.arange(n)
        scale_arr = scale**r
        offset = data[0]*alpha_rev**(r+1)
        pw0 = alpha*alpha_rev**(n-1)

        mult = data*pw0*scale_arr
        cumsums = mult.cumsum()
        out = offset + cumsums*scale_arr[::-1]
        return out
               
    def process_column(self, column_name, initital_T = 100, min_bar = 10, max_bar = 1000):
        init_bar = self.df[:initital_T][column_name].values.tolist()

        ts = [initital_T]
        bts = [bti for bti in self.get_bt(init_bar)]  
        res = []

        buf_bar, vbuf, T = [], [], 0.
        for i in range(initital_T, len(self.df)):

 
            di = self.df.index.values[i]

            buf_bar.append(self.df[column_name].iloc[i])
            bt = self.get_bt(buf_bar)
            theta_t = self.get_theta_t(bt)

            try:
                e_t = self.ewma(np.array(ts), initital_T / 10)[-1]
                e_bt = self.ewma(np.array(bts), initital_T)[-1]
            except:
                e_t = np.mean(ts)
                e_bt = np.mean(bts)
            finally:                   
                if np.isnan(e_bt):
                    e_bt = np.mean(bts[int(len(bts) * 0.9):])
                if np.isnan(e_t):
                    e_t = np.mean(ts[int(len(ts) * 0.9):])

                
            condition = np.abs(theta_t) >= e_t * np.abs(e_bt)

            
            if (condition or len(buf_bar) > max_bar) and len(buf_bar) >= min_bar:

                o = buf_bar[0]
                h = np.max(buf_bar)
                l = np.min(buf_bar)
                c = buf_bar[-1]
                v = np.sum(vbuf)
                
                res.append({
                    self.datetimecolumn: di,
                    'open': o,
                    'high': h,
                    'low': l,
                    'close': c,
                    'volume': v
                })
                
                ts.append(T)
                for b in bt:
                    bts.append(b) 
                    
                buf_bar = []
                vbuf = []
                T = 0.           
            else:
                vbuf.append(self.df[self.volume_column].iloc[i])
                T += 1

        res = pd.DataFrame(res).set_index(self.datetimecolumn)
        return res 
    
    def process_ticks(self, price_column = 'Price', volume_column = 'Size', init = 100, min_bar = 10, max_bar = 1000):
        price_df = self.process_column(price_column, init, min_bar, max_bar)
        return price_df  