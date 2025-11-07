from statsmodels.graphics.tsaplots import acf, pacf
import numpy as np
import pandas as pd
# implement ACF and PACF. The return of the highest function within should be the params for ARIMA later
#check decay rate first, then the significant lags after the first cutoff to diagnose data
# also a figure for plotting is needed or that will be put in front depending on the output 
# of the decay rate. Ignore decay rate if it has been detrended. Detrending again makes no sense
# sanity check the implementation by checking its d parameter with the plotted graph as well as p and q
# based on how you would choose it, this may lead to relaxing th 0.1 boundary for the decay rate a bit

#What params do we need?
# p if AR(p)
# q if MA(q)
# d if differencing is needed 

#How to get the params?
# p and q: implement tran and reeds idea with cutoff threshold and decide upon the right params
# for this to work we need a stationary time series, thus th decay rate as proposed by tran and reed 
# should be implemented first
# this also decided on d
#Thus: first decay and differencing parameter, then p and q

def getARIMA_Params (df_data_series, n_lags, n_alpha, bTrend, bStationary): 
    if not bTrend and not bStationary: 
        #init check
        df_data_series_diffed = df_data_series.diff().dropna()
        t_corr_val_acf, t_conf_int_acf = acf(df_data_series_diffed, nlags=n_lags, alpha=n_alpha)
        t_corr_val_pacf, t_conf_int_pacf = pacf(df_data_series_diffed, nlags=n_lags, alpha=n_alpha)
        n_param_d = getDiffParam(getDecayRate(t_corr_val_acf, t_conf_int_acf), getDecayRate(t_corr_val_pacf, t_conf_int_pacf)) # 2 is the limit to avoid overdifferencing
        if n_param_d == 2:
            print("diff again")
        print(getARMA_Param(t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf, t_conf_int_pacf ))

    elif bTrend and not bStationary:
        print()
        #get params only, decay rate not needed as detrending took care if it
        #if for some reason the decay would still be high, then the tests beforehand did something wrong
    elif not bTrend and bStationary:
        print()
        # well decay can be skipped as well du to it already being stationary

def getDiffParam(n_decay_acf , n_decay_pacf):
    n_param_d = 1 #if a diff param is needed on order of differencing is needed anyway
    if n_decay_acf < 0.1 and n_decay_pacf < 0.1:
        n_param_d_2 = 2
        return n_param_d_2
    else: 
        return n_param_d

def getARMA_Param (t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf,  t_conf_int_pacf):
# Parameter p and q (df_differenced)
# get the minimum cutoff threshold deciding on AR or MA as shown by Reed
#  remember that for case 4 and 5 there is remark 
# on how to choose which model
    bAR = True #if false then MA for the cases in chooseModel()
    print(calcTresholds(t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf,  t_conf_int_pacf))

def calcTresholds(t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf,  t_conf_int_pacf): 
    a_cutoff_T = []
    # print(t_corr_val_acf)
    # print(t_conf_int_acf)

    for k in range (1 , 3):
        # ACF part
        n_upper_band_acf = t_conf_int_acf[k][1] - t_corr_val_acf[k] #get the proper upper limit band
        if t_corr_val_acf[k] == 0 or n_upper_band_acf <= 0:
            n_cut_T_acf = np.inf #to avoid undefined values through math problems
        else: 
            n_cut_T_acf = np.log(n_upper_band_acf) / np.log(abs(t_corr_val_acf[k])) #cutoff threshhold formula by Reed and Tran

        a_cutoff_T.append({
            "Plot_Type": "ACF",
            "Lag": k,
            "Cut_T_Value": n_cut_T_acf
        })

        #PACF part
        n_upper_band_pacf = t_conf_int_pacf[k][1] - t_corr_val_pacf[k] #get the proper upper limit band
        if t_corr_val_pacf[k] == 0 or n_upper_band_pacf <= 0:
            n_cut_T_pacf = np.inf
        else:
            n_cut_T_pacf = np.log(n_upper_band_pacf) / np.log(abs(t_corr_val_pacf[k]))

        a_cutoff_T.append({
            "Plot_Type": "PACF",
            "Lag": k,
            "Cut_T_Value": n_cut_T_pacf
        })
    return pd.DataFrame(a_cutoff_T)

def getDecayRate(t_corr_val,t_conf_int):
    # cutoff rate as shown by Tran and Reed 
    n_cutoff = getLagBefCutoff(t_corr_val,t_conf_int)
    a_acf = t_corr_val
    n_sum_rate_of_change = 0

    for k in range (0, n_cutoff):
        nominator = abs(a_acf[k]) - abs(a_acf[k + 1])
        denominator = abs(a_acf[k])
        n_ratio = nominator / denominator
        n_sum_rate_of_change = n_sum_rate_of_change + n_ratio

    return n_sum_rate_of_change / n_cutoff

def getLagBefCutoff(t_corr_val,t_conf_int): 
    # Get parameter M for decay by checking which lag is the last before it dips below the band
    # should be usable for both ACF and PACF
    # Returns M
    # print(t_corr_val)
    # print(t_conf_int)

    # Get the index upper band (margin of error) value by UCI - r_k (value at lag k)
    #   check with abs(r_k) <= margin_of_error

    for k in range(1, len(t_corr_val)): #start at index 1, 0 would make no sense as a value is always autocorrelated with itself
        n_r_k = t_corr_val[k] #autocorrelation at lag k
        n_uci_k = t_conf_int[k][1] #gets upper confidence interval

        n_margin_err = n_uci_k - n_r_k
        if abs(n_r_k) <= n_margin_err:
            return k -1 
    
    return len(t_corr_val) - 1


# getLagBefCutoff -> decay rate -> 