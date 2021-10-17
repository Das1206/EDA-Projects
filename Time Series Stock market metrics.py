import pandas as pd 
import numpy as np 
import statsmodels.api as sm
from pandas.tseries.offsets import DateOffset
import warnings
warnings.filterwarnings('ignore')

# change the directory here, you will find the output csv  in the same location  
df =  pd.read_csv(r'D:\Downloads\Tickers_For_Algo_Training.csv')
df = df[::-1]
df['Green'] = df['Positive_Directional_Index_Indicator']
df['Red'] = df['Negative_Directional_Index_Indicator']
df['Blue'] = df['ADX_Indicator']
fdf = pd.DataFrame()

for ticker in df.Ticker.unique():
    aadf = df[df['Ticker']== ticker]
    end = aadf['Ticker'].count()
    start = end-6
    ends = end+32
    ends1 = ends + 1

    greendf = aadf[['Date','Green']]
    bluedf = aadf[['Date','Blue']]
    reddf = aadf[['Date','Red']]

    greendf['Date']=pd.to_datetime(greendf['Date'])
    bluedf['Date']=pd.to_datetime(bluedf['Date'])
    reddf['Date']=pd.to_datetime(reddf['Date'])

    greendf.set_index('Date',inplace=True)
    bluedf.set_index('Date',inplace=True)
    reddf.set_index('Date',inplace=True)

    # green 
    modelg=sm.tsa.statespace.SARIMAX(greendf['Green'],order=(1, 1, 1),seasonal_order=(1,1,1,30))
    resultsg=modelg.fit()
    future_datesg=[greendf.index[-1]+ DateOffset(days=x)for x in range(0,60)]
    future_datest_dfg=pd.DataFrame(index=future_datesg[1:],columns=greendf.columns)
    future_dfg=pd.concat([greendf,future_datest_dfg])
    forecastg = resultsg.predict(start = start, end = ends, dynamic= True) 
    forecastg.index = future_dfg[start:ends1].index
    
    # red
    modelr=sm.tsa.statespace.SARIMAX(reddf['Red'],order=(1, 1, 1),seasonal_order=(1,1,1,30))
    resultsr=modelr.fit()
    future_datesr=[reddf.index[-1]+ DateOffset(days=x)for x in range(0,60)]
    future_datest_dfr=pd.DataFrame(index=future_datesr[1:],columns=reddf.columns)
    future_dfr=pd.concat([reddf,future_datest_dfr])
    forecastr = resultsr.predict(start = start, end = ends, dynamic= True) 
    forecastr.index = future_dfr[start:ends1].index

    # blue
    modelb=sm.tsa.statespace.SARIMAX(bluedf['Blue'],order=(1, 1, 1),seasonal_order=(1,1,1,30))
    resultsb=modelb.fit()
    future_datesb=[bluedf.index[-1]+ DateOffset(days=x)for x in range(0,60)]
    future_datest_dfb=pd.DataFrame(index=future_datesb[1:],columns=bluedf.columns)
    future_dfb=pd.concat([bluedf,future_datest_dfb])
    forecastb = resultsb.predict(start = start, end = ends, dynamic= True) 
    forecastb.index = future_dfb[start:ends1].index
    
    tempdf = pd.concat((forecastg.rename('Green Forecast'),forecastr.rename('Red Forecast'),forecastb.rename('Blue Forecast')), axis = 1)
    tempdf['Ticker'] = ticker
    fdf = pd.concat([tempdf,fdf], axis = 0)
#print(fdf)


def wrapper_days(tempdf,upper,lower):
    for i in ['Red Forecast']:
        tempdf = tempdf[::-1]
        j = 0
        overforty = 0
        ls = []
        count = 0 
        for k in tempdf[i]:
            
            if k > lower:
                count += 1 
        if count > 0:     
            while j < len(tempdf[i]):

                if float(tempdf[i][j]) < lower:
                    overforty +=1
                    ls.append(overforty)
                else:
                    overforty = 0
                    ls.append(overforty)
                j = j + 1
            #print(ls)
            tempdf[i+' until AL days'] = ls
        else:
            tempdf[i+' until AL days'] = 'NaN'
        
        
    # Above uPPER for RED < 25
    for i in ['Red Forecast']:
        tempdf = tempdf[::-1]
        j = 0
        overforty = 0
        ls = []
        count = 0 
        for k in tempdf[i]:
            if k > upper:
                count += 1 
        if count > 0:    
            while j < len(tempdf[i]):

                if float(tempdf[i][j]) < upper:
                    overforty +=1
                    ls.append(overforty)
                else:
                    overforty = 0
                    ls.append(overforty)
                j = j + 1
            #print(ls)
            tempdf[i+' until AU days'] = ls
        else:
            tempdf[i+' until AU days'] = 'NaN'
            
    # Blue AND GREEN below upper > 40 
    for i in ['Blue Forecast','Green Forecast']:
        tempdf = tempdf[::-1]
        j = 0
        overforty = 0
        ls = []
        count = 0 
        for k in tempdf[i]:
            if k < upper:
                count += 1 
        if count > 0:    
            while j < len(tempdf[i]):

                if float(tempdf[i][j]) > upper:
                    overforty +=1
                    ls.append(overforty)
                else:
                    overforty = 0
                    ls.append(overforty)
                j = j + 1
            #print(ls)
            tempdf[i+' until BU days'] = ls
        else:
            tempdf[i+' until BU days'] = 'NaN'
            
    # Blue AND GREEN below lOWER > 40 
    for i in ['Blue Forecast','Green Forecast']:
        tempdf = tempdf[::-1]
        j = 0
        overforty = 0
        count = 0
        ls = []
        for k in tempdf[i]:
            if k < lower:
                count += 1
        if count > 0:
            while j < len(tempdf[i]):

                if float(tempdf[i][j]) > lower:
                    overforty +=1
                    ls.append(overforty)
                else:
                    overforty = 0
                    ls.append(overforty)
                j = j + 1
            #print(ls)
            tempdf[i+' until BL days'] = ls
        else: 
            tempdf[i+' until BL days'] = 'NaN'
    return tempdf

def lastfunc(df,fdf):
    ffdf = pd.DataFrame()
    for ticker in fdf.Ticker.unique():
        pdf = df[df.Ticker == ticker]
        lower = pdf['Min_Avg_Low'].mean()
        upper = pdf['Max_Avg_Upper'].mean()
        tempdf = fdf[fdf['Ticker']== ticker]
        ffdf = pd.concat([wrapper_days(tempdf,upper,lower),ffdf], axis = 0)
    return ffdf

tfdf = lastfunc(df,fdf)
#print(tfdf.head())
path = 'D:/Downloads/'
tfdf.to_csv(path+'predictedtickers.csv')