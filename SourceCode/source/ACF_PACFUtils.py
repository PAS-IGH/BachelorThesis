from statsmodels.graphics.tsaplots import acf, pacf
import numpy as np
import pandas as pd
import math

def getARIMA_Params (df_data_series, n_lags, n_alpha, dict_stat, dict_results, df_trend_series = None):

    """
    Returns the estimated parameters needed for ARIMA modelling based on the stationary type.
    This function performs the following operations:
        1. Decides based on the provided stationary type how to estimate the parameters
        2. Gets the acf and pacf values as well as the confidence intervalls based on Barletts formula
        3. Obtains the differencing parameter based on the computed decay rate of the acf and pacf
        4. Gets the p and q params by applying Tran and Reeds method
        5. Returns ARIMA(p,d,q) parameters
    Args:
        df_data_series (pandas.DataSeries): A series which ARIMA modelling is applied to
        n_lags (int): An integer dictating the number of lags for acf and pacf
        n_alpha (float): A floating point number indicating the alpha value
        dict_stat (dict): A dictionary containing information regrarding stationary type
        dict_results (dict): A dictionary containing information of previous time series analysis steps
        df_trend_series(pandas.DataSeries): A series containing the trend values for detrending
    Returns:
        tuple (p,d,q):
            p(int): AR(p) parameter;
            d(int): Differencing parameter;
            q(int): MA(q) parameter;
    """

    if dict_stat["b_Difference"]:  
        #init check
        df_data_series_diffed = df_data_series.diff().dropna()
        t_corr_val_acf, t_conf_int_acf = acf(df_data_series_diffed, nlags=n_lags, alpha=n_alpha)
        t_corr_val_pacf, t_conf_int_pacf = pacf(df_data_series_diffed, nlags=n_lags, alpha=n_alpha)
        n_param_d = getDiffParam(getDecayRate(t_corr_val_acf, t_conf_int_acf), getDecayRate(t_corr_val_pacf, t_conf_int_pacf))
        
        #====== 2 is the limit to avoid overdifferencing====
        if n_param_d == 2:
            df_data_series_diffed = df_data_series_diffed.diff().dropna()
            t_corr_val_acf, t_conf_int_acf = acf(df_data_series_diffed, nlags=n_lags, alpha=n_alpha)
            t_corr_val_pacf, t_conf_int_pacf = pacf(df_data_series_diffed, nlags=n_lags, alpha=n_alpha)
        #============================================================

        p, q = getARMA_Param(t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf, t_conf_int_pacf, dict_results)

    elif dict_stat["b_Detrend"]:
        #get params only, decay rate not needed as detrending took care if it
        n_param_d = 0
        df_data_series_detrended = df_data_series.iloc[:,0] - df_trend_series
        t_corr_val_acf, t_conf_int_acf = acf(df_data_series_detrended, nlags=n_lags, alpha=n_alpha)
        t_corr_val_pacf, t_conf_int_pacf = pacf(df_data_series_detrended, nlags=n_lags, alpha=n_alpha)
        p, q = getARMA_Param(t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf, t_conf_int_pacf, dict_results)

    elif dict_stat["b_Stationary"]:
        # well decay can be skipped as well due to it already being stationary
        n_param_d = 0
        t_corr_val_acf, t_conf_int_acf = acf(df_data_series, nlags=n_lags, alpha=n_alpha)
        t_corr_val_pacf, t_conf_int_pacf = pacf(df_data_series, nlags=n_lags, alpha=n_alpha)
        p, q = getARMA_Param(t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf, t_conf_int_pacf, dict_results)

    # === Save Results =========================================================
    dict_results["decay_rate_acf"] = getDecayRate(t_corr_val_acf, t_conf_int_acf)
    dict_results["decay_rate_pacf"] = getDecayRate(t_corr_val_pacf, t_conf_int_pacf)
    dict_results["ARIMA_Params_estimated"] = {
        "p": p,
        "d": n_param_d,
        "q": q
    }
    # ==========================================================================
    return p, n_param_d , q


def getDiffParam(n_decay_acf , n_decay_pacf):

    """
    Obtains the differencing parameter based on the decay rate.
    In order to avoid overdifferencing the series is differenced up to a maximum of two times.
    This function performs the following operations:
        1. Sets the initial value for the differencing parameter to one, as this method is called only in a differencing case
        2. Computes the decay rates of both acf and pacf, followng Tran and Reed's method and increases the value of the differencing parameter accordingly
        3. Returns the differencing parameter
    Args:
        n_decay_acf (float): The decay rate of the acf plot
        n_decay_pacf(float): The decay rate of the pacf plot
    Returns:
        n_param_d(int), n_param_d_2(int): Differencing parameter of order one and two
    """
    n_param_d = 1 
    if n_decay_acf < 0.1 and n_decay_pacf < 0.1:
        n_param_d_2 = 2
        return n_param_d_2
    else: 
        return n_param_d

def getARMA_Param (t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf,  t_conf_int_pacf, dict_results):
    """
    Wrapper function respoonsible for passing on the results of the minimum cutoff threshold computation to the parameter estimation following Tran and Reed's approach.
    This function performs the following operations:
        1. Applies the calcThreshold() function to get cutoff threshold values
        2. Applies the getMinLagThresholds() function with the previous result to get the minimum cutoff threshold
        3. Applies get_p_q() function with the previous result to estimate p and q parameters for ARMA(p,q)
    Args:
        t_corr_val_acf(numpy.array): Array containing the acf values
        t_conf_int_acf(numpy.array): Array containing the acf confidence intervals 
        t_corr_val_pacf(numpy.array): Array containing the pacf values
        t_conf_int_pacf(numpy.array): Array containing the pacf confidence intervals
        dict_results (dict): A dictionary containing information of previous time series analysis steps
    Returns:
        tuple(p,q): 
            p(int): AR(p) parameter;
            q(int): MA(q) parameter; 
    """
    return get_p_q(getMinLagThresholds(calcTresholds(t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf,  t_conf_int_pacf), dict_results))


def get_p_q (df_minLag) :
    """
    Estimates the p and q parameters based on the minimum cutoff threshold, following Tran and Reed's approach.
    This function performs the following operations:
        1. Obtains the correct model type based on plot type
        2. Acquires the cutoff threshold values as well as their rounded down counterparts, from the minimum cutoff threshold
        3. Returns a tuple with the p and q params, determined by Tran and Reed's approach
    Args:
        df_minLag (pandas.DataFrame): DataFrame containing lags, cutoff values and plot type as derived by the minimum cutoff lag
    Returns:
        tuple(p,q): 
            p(int): AR(p) parameter;
            q(int): MA(q) parameter; 
    """
    if df_minLag["Plot_Type"][0] == "ACF":
        b_AR = False
    elif df_minLag["Plot_Type"][0] == "PACF":
        b_AR = True

    n_lag1_val = df_minLag.loc[df_minLag["Lag"] == 1, "Cut_T_Value"].iloc[0]
    n_lag1_val_rounded = math.floor(n_lag1_val) # always rounded down as said by Tran and Reed. Keep the original values for special cases 4 and 5
    n_lag2_val = df_minLag.loc[df_minLag["Lag"] == 2, "Cut_T_Value"].iloc[0]
    n_lag2_val_rounded= math.floor(n_lag2_val)

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
        
        if n_lag1_val_rounded > 1:
            if b_AR:
                return 2 , 0
            else:
                return 0 , 2

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


def getMinLagThresholds(df_thresholds, dict_results):
    """
    Creates a dataframe containing the plot type,lags and cutoff values derived by determining the minimum cutoff threshold.
    This function performs the following operations:
        1. Locates the row of the minimum cutoff threshold based on the given set of cutoff thresholds
        2. Identifies its plot type (ACF or PACF)
        3. Filters the given cutoff thresholds by their plot type and returns the filtered set
    Args:
        df_thresholds(pandas.DataFrame): A dataframe containing cutoff thresholds up to lag two for both plot types (ACF, PACF)
    Returns:
        df_filtered_lags(pandas.DataFrame): Filtered dataframe for estimating p and q values following Tran and Reed's approach
    """
    n_min_row = df_thresholds.loc[df_thresholds["Cut_T_Value"].idxmin()] #gets the row with the minimum cutoff threshold
    s_plot_type = n_min_row["Plot_Type"]
    df_filtered_lags = df_thresholds[df_thresholds["Plot_Type"] == s_plot_type].reset_index(drop=True)

    # === Save Results ===========================================
    dict_results["cutoff_thresholds"] = df_thresholds
    dict_results["cutoff_thresholds_minimum"] = n_min_row
    dict_results["cutoff_thresholds_filtered"] = df_filtered_lags
    # ============================================================
    return df_filtered_lags

def calcTresholds(t_corr_val_acf, t_conf_int_acf, t_corr_val_pacf,  t_conf_int_pacf): 
    """
    Calculates cutoff thresholds based on the formula given by Tran and Reed.
    This function performs the following operations:
        1. Calculates the proper upper limit (The blue band)
        2. Employs the cutoff threshold formula
        3. Appends the result as a dictionary to a list 
        4. Returns the list of cutoff thresholds in the form of a dataframe 
    Args:
        t_corr_val_acf(numpy.array): Array containing the acf values
        t_conf_int_acf(numpy.array): Array containing the acf confidence intervals 
        t_corr_val_pacf(numpy.array): Array containing the pacf values
        t_conf_int_pacf(numpy.array): Array containing the pacf confidence intervals
    Returns:
        (pandas.DataFrame): A dataframe with the cutoff thresholds up to lag two for both ACF and PACF
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
            n_cut_T_pacf = np.log(n_upper_band_pacf) / np.log(abs(t_corr_val_pacf[k])) #cutoff threshhold formula by Reed and Tran

        a_cutoff_T.append({
            "Plot_Type": "PACF",
            "Lag": k,
            "Cut_T_Value": n_cut_T_pacf
        })
    return pd.DataFrame(a_cutoff_T)

def getDecayRate(t_corr_val,t_conf_int):
    """
    Calculates the decay rate based on the formula provided by Tran and Reed.
    This function performs the following operations:
        1. Gets the lag before correlation becomes insignificant
        2. Applies Tran and Reed's decay rate formula
    Args:
        t_corr_val(numpy.array): Array containing the autocorrelation values
        t_conf_int(numpy.array): Array containing the autocorrelation confidence intervals
    Returns:
        (float): The decay rate
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
    Obtains the lag before subsequent lags become insignificant.
    This function performs the following operations:
        1. Gets the upper confidence interval and correlation value at a lag k
        2. Subtracts the correlation value from the upper confidence interval at a lag k
        3. As soon as the resulting value falls below the margin of error the previous lag (ergo k - 1) is selected and returned
    Args:
        t_corr_val(numpy.array): Array containing the autocorrelation values
        t_conf_int(numpy.array): Array containing the autocorrelation confidence intervals
    Returns:
        (int): The cutoff lag 
    """
    for k in range(1, len(t_corr_val)): #start at index 1, 0 would make no sense as a value is always autocorrelated with itself
        n_r_k = t_corr_val[k] #autocorrelation at lag k
        n_uci_k = t_conf_int[k][1] #gets upper confidence interval

        n_margin_err = n_uci_k - n_r_k
        if abs(n_r_k) <= n_margin_err:
            return k -1 
    
    return len(t_corr_val) - 1