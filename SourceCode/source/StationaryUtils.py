import pandas as pd
from statsmodels.tsa.stattools import adfuller as adf

def getStationary(bTrend, df_DataSeries):

    # implement AIC to get the beste possible regression here
    if bTrend: #if Trend exists then RWWWD with DT ADF, so do constant and trend param

        t_adf = adf(df_DataSeries, regression ='ct', autolag="AIC") #get the test statistic and critical value
        adf_stat= t_adf[0]
        adf_crit_005alpha = t_adf[4][1]
        bNullHypo = True

        if adf_stat < adf_crit_005alpha: #reject null hypotheses; thus there is a DT; only detrend the series
            bNullHypo = False
        else:
            bNullHypo = True #failure to reject RWWD; thus difference

        return 
    
    if not bTrend:
        return adf(df_DataSeries, regression ='c', autolag="AIC")