from statsmodels.graphics.tsaplots import acf, pacf
import numpy as np
import pandas as pd
import math
# implement ACF and PACF. The return of the highest function within should be the params for ARIMA later
#check decay rate first, then the significant lags after the first cutoff to diagnose data
# also a figure for plotting is needed or that will be put in front depending on the output 
# of the decay rate. Ignore decay rate if it has been detrended. Detrending again makes no sense
# sanity check the implementation by checking its d parameter with the plotted graph as well as p and q
# based on how you would choose it, this may lead to relaxing th 0.1 boundary for the decay rate a bit

#How to get the params?
# p and q: implement tran and reeds idea with cutoff threshold and decide upon the right params
# for this to work we need a stationary time series, thus th decay rate as proposed by tran and reed 
# should be implemented first
# this also decided on d
#Thus: first decay and differencing parameter, then p and q

def getARIMA_Params (df_data_series, n_lags, n_alpha, bTrend, bStationary, df_trend_series = None):

    """
    Returns the parameter needed for performing ARIMA operations depending on if the data needs to be
    differenced or detrended or is already stationary
    This function performs the following operations:
        1. Takes the parameters and decides upon the booleans which path to follow
        2. Get the acf and pacf values as well as the confidence intervalls based on Barletts formula
        3. Decide on the d param by looking at the respective decay rates (In differencing case; otherwise its 0)
            a. difference again if the result is that differecing once might not be enough
        4. Get the p and q params by applying Trand and Reeds method
        5. returns p, d, q
    Args:
        df_data_series (pandas.DataSeries): A series where the params are needed for ARIMA
        n_lags (int): An integer dictating the number of lags for acf and pacf
        n_alpha (float): A floating point number indicating the alpha value
        bTrend (bool): A boolean for indicating a trend
        bStationary(bool): A boolean indicating stationary
        df_trend_series(pandas.DataSeries): A series containing the trend values for detrending
    Returns:
        A tuple containing (p, n_param_d, q) ergo (p, d, q) for ARIMA(p,d,q)
    """

    if not bTrend and not bStationary: 
        #init check
        df_data_series_diffed = df_data_series.diff().dropna()
        t_corr_val_acf, t_conf_int_acf = acf(df_data_series_diffed, nlags=n_lags, alpha=n_alpha)
        t_corr_val_pacf, t_conf_int_pacf = pacf(df_data_series_diffed, nlags=n_lags, alpha=n_alpha)
        n_param_d = getDiffParam(getDecayRate(t_corr_val_acf, t_conf_int_acf), getDecayRate(t_corr_val_pacf, t_conf_int_pacf)) # 2 is the limit to avoid overdifferencing
        if n_param_d == 2:
            print("diff again")
        p, q = getARMA_Param(t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf, t_conf_int_pacf)
        return p, n_param_d , q

    elif bTrend and not bStationary:
        #get params only, decay rate not needed as detrending took care if it
        n_param_d = 0
        df_data_series_detrended = df_data_series - df_trend_series
        t_corr_val_acf, t_conf_int_acf = acf(df_data_series_detrended, nlags=n_lags, alpha=n_alpha)
        t_corr_val_pacf, t_conf_int_pacf = pacf(df_data_series_detrended, nlags=n_lags, alpha=n_alpha)
        p, q = getARMA_Param(t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf, t_conf_int_pacf)
        return p, n_param_d , q
    elif not bTrend and bStationary:
        # well decay can be skipped as well due to it already being stationary
        n_param_d = 0
        t_corr_val_acf, t_conf_int_acf = acf(df_data_series, nlags=n_lags, alpha=n_alpha)
        t_corr_val_pacf, t_conf_int_pacf = pacf(df_data_series, nlags=n_lags, alpha=n_alpha)
        p, q = getARMA_Param(t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf, t_conf_int_pacf)
        return p, n_param_d , q


def getDiffParam(n_decay_acf , n_decay_pacf):

    """
    Gets the d param based on the provided decay rate. To avoid overdifferencing a maximum of n_param_d_2 = 2 is introduced
    This operations performs as follows:
        1. Sets n_param_d to one because the mehod is used when differencing is already needed
        2. Checks if both decay rate are less than 10 percent (as dictated by Tran and Reed)
    Args:
        n_decay_acf (float): The decay rate of the acf plot
        n_decay_pacf(float): The decay rate of the pacf plot
    Returns:
        The order of differencing
        n_param_d or n_param_d_2
    """
    n_param_d = 1 #if a diff param is needed on order of differencing is needed anyway
    if n_decay_acf < 0.1 and n_decay_pacf < 0.1:
        n_param_d_2 = 2
        return n_param_d_2
    else: 
        return n_param_d

def getARMA_Param (t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf,  t_conf_int_pacf):
    """
    Wrapper methods containing a pipeline where the results of the minimum cutoff Treshold is used to estimate possible p and q params for ARIMA
    This method performs the following operations:
        1. Returns the result of the pipeline  calcTresholds->getMinLagThresholds->get_p_q
    Args:
        t_corr_val_acf(numpy.array): Array containing the acf values
        t_conf_int_acf(numpy.array): Array containing the acf confidence intervals 
        t_corr_val_pacf(numpy.array): Array containing the pacf values
        t_conf_int_pacf(numpy.array): Array containing the pacf confidence intervals
    Returns:
        p and q values for ARIMA operations
    """
    return get_p_q(getMinLagThresholds(calcTresholds(t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf,  t_conf_int_pacf)))


def get_p_q (df_minLag) :
    """
    Estimates the p and q params based on the deriviation provided by Tran and Reed 
    This method performs the following operations:
        1. Checks the plot type estimated to be the right on provided by the parameter; look at getMinLagThresholds() for that
        2. Get the cut treshold values and the rounded down integer form of it
        3. Check the cases as the stated by the Tran and Reed table
        4. Return a tuple containing the estimated p and q
    Args:
        df_minLag (pandas.DataFrame): DataFrame contaning the the lag, plot type and, cutoff threshhold value based on the minimum cutoff threshhold
    Returns:
        p and q values for ARIMA operations
    """
    if df_minLag["Plot_Type"][0] == "ACF":
        b_AR = False
    elif df_minLag["Plot_Type"][0] == "PACF":
        b_AR = True

    n_lag1_val = df_minLag.loc[df_minLag["Lag"] == 1, "Cut_T_Value"].iloc[0]
    n_lag1_val_rounded = math.floor(n_lag1_val) # always rounded down as said by Tran and Reed. Keep the original values for special cases 4 and 5
    n_lag2_val = df_minLag.loc[df_minLag["Lag"] == 2, "Cut_T_Value"].iloc[0]
    n_lag2_val_rounded= math.floor(n_lag2_val)
    print(f"lag 1: {n_lag1_val} lag 2:{n_lag2_val}")

    #case 0
    if n_lag2_val_rounded == 0 and n_lag1_val_rounded == 0: 
        return 0 , 0
    elif (
        (n_lag2_val_rounded == 0 and n_lag1_val_rounded == 1) or #case 1
        (n_lag2_val_rounded == 0 and n_lag1_val_rounded > 1) or #case 2
        (n_lag2_val_rounded == 1 and n_lag1_val_rounded == 0) or #case 3
        (n_lag2_val_rounded > 1 and n_lag1_val_rounded == 0) or #case 6
        (n_lag2_val_rounded > 1 and n_lag1_val_rounded == 1) #case 7
    ):
        if b_AR:
            return 1 , 0
        else:
            return 0 , 1
     #special case 4; see Tran and Reed for detailed info
    elif n_lag2_val_rounded == 1 and n_lag1_val_rounded == 1:
        if n_lag1_val < n_lag2_val:
            if b_AR:
                return 1 , 0
            else:
                return 0 , 1
        elif n_lag1_val > n_lag2_val and n_lag2_val > 1.5: 
            if b_AR:
                return 2 , 0
            else:
                return 0 , 2
        else:
            if b_AR:
                return 1 , 0
            else:
                return 0 , 1
    #special case 5; see Tran and Reed for detailed info
    elif n_lag2_val_rounded == 1 and n_lag1_val_rounded > 1:              
        if n_lag1_val > n_lag2_val and n_lag2_val > 1.5:
            if b_AR:
                return 2 , 0
            else:
                return 0 , 2
        else:
            if b_AR:
                return 1 , 0
            else:
                return 0 , 1
    # case 8 ARMA case
    elif n_lag2_val_rounded > 1 and n_lag1_val_rounded > 1: 
        return 1,1



def getMinLagThresholds(df_thresholds):
    """
    Creates a dataframe containing the plot type and lags based on the minimum threshold value
    This method performs the following operations:
        1. Takes the calculated thresholds and locates the one with the lowest cutoff threshold value
        2. Extracts plot type (ACF or PACF)
        3. Filters the given thresholds down based on the plot type and returns the frame
    Args:
        df_thresholds(pandas.DataFrame): A dataframe containing cutoff thresholds up to lag 2 for both plot types
    Returns:
        Filtered dataframe to be used for estimating p and q values
    """
    n_min_row = df_thresholds.loc[df_thresholds["Cut_T_Value"].idxmin()] #gets the row with the minimum cutoff threshold
    s_plot_type = n_min_row["Plot_Type"]
    df_filtered_lags = df_thresholds[df_thresholds["Plot_Type"] == s_plot_type]
    print(df_filtered_lags)
    return df_filtered_lags

def calcTresholds(t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf,  t_conf_int_pacf): 
    """
    Calculates cutoff thresholds based on the formula given by Tran and Reed
    This method performs the following operations:
        1. Calculates the proper upper limit (The blue band)
        2. Employs the cutoff threshold formula
        3. Appends the result as a dictionary to a list
    Args:
        t_corr_val_acf(numpy.array): Array containing the acf values
        t_conf_int_acf(numpy.array): Array containing the acf confidence intervals 
        t_corr_val_pacf(numpy.array): Array containing the pacf values
        t_conf_int_pacf(numpy.array): Array containing the pacf confidence intervals
    Returns:
        A dataframe with the cutoff thresholds up to lag 2 for both ACF and PACF
    """
    a_cutoff_T = []
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
    """
    Calculates the decay rate based on the formula provided by Tran and Reed
    This method performs the following operations:
        1. Gets the lag before the next lag becomes insignificant
        2. Performs the summation of the fraction
        3. Divides by the cutoff lag 
    Args:
        t_corr_val(numpy.array): Array containing the autocorrelation values
        t_conf_int(numpy.array): Array containing the autocorrelation confidence intervals
    Returns:
        The decay rate 
    """

    n_cutoff = getLagBefCutoff(t_corr_val,t_conf_int)
    n_sum_rate_of_change = 0

    for k in range (0, n_cutoff):
        nominator = abs(t_corr_val[k]) - abs(t_corr_val[k + 1])
        denominator = abs(t_corr_val[k])
        n_ratio = nominator / denominator
        n_sum_rate_of_change = n_sum_rate_of_change + n_ratio

    return n_sum_rate_of_change / n_cutoff

def getLagBefCutoff(t_corr_val,t_conf_int): 
    """
    Get the lag before subsequent lags become insignificant.
    This method performs the following operations:
        1. Gets the upper confidence interval and value at lag k
        2. Subtracts upper confidence interval from value at lag k
        3. As soon as the value at lag k is less than the margin of error the lag before is returned
    Args:
        t_corr_val(numpy.array): Array containing the autocorrelation values
        t_conf_int(numpy.array): Array containing the autocorrelation confidence intervals
    Returns:
        The cutoff lag  

    """
    for k in range(1, len(t_corr_val)): #start at index 1, 0 would make no sense as a value is always autocorrelated with itself
        n_r_k = t_corr_val[k] #autocorrelation at lag k
        n_uci_k = t_conf_int[k][1] #gets upper confidence interval

        n_margin_err = n_uci_k - n_r_k
        if abs(n_r_k) <= n_margin_err:
            return k -1 
    
    return len(t_corr_val) - 1