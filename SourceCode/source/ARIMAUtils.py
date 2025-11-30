
"""
Provides functions for selecting the optimal model with a grid search based on estimated ARIMA parameters.
"""
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from coreforecast.scalers import inv_boxcox
from statsmodels.tsa.forecasting.stl import STLForecast

def getOptimalModel(df_series, p, d, q, dict_results):
    """
    A Wrapper function for obtaining the optimal ARIMA model.
    This function performs the following operations:
        1. Chooses a model selection implementation based on the given p and q parameters
    Args:
        df_series (pandas.DataSeries): The data upon which the model is supposed to be fitted
        p (int): An integer dictating the AR part of ARIMA
        d (int): An integer dictating the differencing part of ARIMA
        q (int): An integer dictating the MA part of ARIMA
    Returns:
        fitted_model (statsmodels.tsa.arima.model.ARIMAResults): The fitted optimal model
    """
    if p > 0 and q == 0:
        fitted_model = fitAR(df_series, p, d, dict_results)["model"]
    elif p == 0 and q > 0:
        fitted_model = fitMA(df_series,d, q, dict_results)["model"]
    else:
        fitted_model = fitARIMA(df_series, p, d, q, dict_results)["model"]
    
    return fitted_model


def fitAR(df_series, p, d, dict_results):
    """
    Obtains the optimal model based on the given AR parameter and Ljung-Box test.
    This function performs the following operations:
        1. Builds a range based on the estimated parameters ranging from one below and two above the estimtation (range end is exclusive)
        2. Qualifies a model based on its Ljung-Box p-value
        3. Adds the model along order, aic and Ljung-Box p-value to a list if it qualifies 
        4. Compares the AIC scores of the models within the list to determine the optimal one and return it
    Args:
        df_series (pandas.DataSeries): The data upon which the model is supposed to be fitted
        p (int): An integer dictating the AR part of ARIMA
        d (int): An integer dictating the differencing part of ARIMA
    Returns:
        best_model (statsmodels.tsa.arima.model.ARIMAResults): The fitted optimal AR model
    """
    p_range = range(max(0, p - 1), p + 3)
    good_models = [] 
    dict_results["models"] = []
    for i in p_range:
        if i==0:
            continue
        try:
            if dict_results['stationary_status']["stat_type"] == "trend":
                fitted_model = ARIMA(df_series, order=(i , d, 0), trend="ct").fit()
            else:
                fitted_model = ARIMA(df_series, order=(i , d, 0)).fit()
            
            n_ljungBox_results = fitted_model.test_serial_correlation(method="ljungbox")
            n_ljungBox_pValue = n_ljungBox_results[0,1,-1] #gets the p-value of the last lag, portmonteau cumulative test

            # === Save Results ========
            dict_results["models"].append({
                "order": (i, d, 0),
                "aic": fitted_model.aic,
                "model": fitted_model,
                "ljung_box_pValue" : n_ljungBox_pValue 
            })
            # ==========================
            if n_ljungBox_pValue > 0.05: 
                good_models.append({
                    "order": (i, d, 0),
                    "aic": fitted_model.aic,
                    "model": fitted_model,
                    "ljung_box_pValue" : n_ljungBox_pValue, 
                })
            else:
                print("one lag is not good enough")
        except:
            continue
    if not good_models:
        print("nothing found AR")
    else:
        best_model = min(good_models, key=lambda x: x["aic"]) #lowest AIC score
        return best_model
    
def fitMA(df_series,d, q, dict_results):

    """
    Obtains the optimal model based on the given MA parameter and Ljung-Box test.
    This function performs the following operations:
        1. Builds a range based on the estimated parameters ranging from one below and two above the estimtation (range end is exclusive)
        2. Qualifies a model based on its Ljung-Box p-value
        3. Adds the model along order, aic and Ljung-Box p-value to a list if it qualifies 
        4. Compares the AIC scores of the models within the list to determine the optimal one and return it
    Args:
        df_series (pandas.DataSeries): The data upon which the model is supposed to be fitted
        q (int): An integer dictating the MA part of ARIMA
        d (int): An integer dictating the differencing part of ARIMA
    Returns:
        best_model (statsmodels.tsa.arima.model.ARIMAResults): The fitted optimal MA model
    """
    q_range = range(max(0, q - 1), q + 3)
    good_models = [] 
    dict_results["models"] = []
    for i in q_range:
        if i==0:
            continue
        try:
            if dict_results['stationary_status']["stat_type"] == "trend":
                fitted_model = ARIMA(df_series, order=(0 , d, i), trend="ct").fit()
            else:
                fitted_model = ARIMA(df_series, order=(0 , d, i)).fit()

            n_ljungBox_results = fitted_model.test_serial_correlation(method="ljungbox")
            n_ljungBox_pValue = n_ljungBox_results[0,1,-1] #gets the p-value of the last lag, portmonteau cumulative test

            # === Save Results ========
            dict_results["models"].append({
                "order": (0 , d , i),
                "aic": fitted_model.aic,
                "model": fitted_model,
                "ljung_box_pValue" : n_ljungBox_pValue  
            })
            # ==========================

            if n_ljungBox_pValue > 0.05: 
                good_models.append({
                    "order": (0 , d , i),
                    "aic": fitted_model.aic,
                    "model": fitted_model,
                    "ljung_box_pValue" : n_ljungBox_pValue   
                })
            else:
                print("one lag is not good enough")
        except Exception as e:
            print(e)
            continue
    if not good_models:
        print("nothing found MA")
    else:
        best_model = min(good_models, key=lambda x: x["aic"])# lowest AIC score wins
        return best_model
    
def fitARIMA(df_series,p, d, q, dict_results):

    """
    Obtains the optimal model based on the given ARIMA parameters and Ljung-Box test.
    This function performs the following operations:
        1. Builds a range based on the estimated parameters ranging from one below and two above the estimtation (range end is exclusive)
        2. Qualifies a model based on its Ljung-Box p-value
        3. Adds the model along order, aic and Ljung-Box p-value to a list if it qualifies 
        4. Compares the AIC scores of the models within the list to determine the optimal one and return it
    Args:
        df_series (pandas.DataSeries): The data upon which the model is supposed to be fitted
        p (int): An integer dictating the AR part of ARIMA
        d (int): An integer dictating the differencing part of ARIMA
        q (int): An integer dictating the MA part of ARIMA
    Returns:
        best_model (statsmodels.tsa.arima.model.ARIMAResults): The fitted optimal ARIMA model
    """
    p_range = range(max(0, p - 1), p + 3) 
    q_range = range(max(0, q - 1), q + 3)
    good_models = [] #models whose ljung box pvalue ar not below 0.05
    dict_results["models"] = []
    for i in p_range:
        for j in q_range:
            if i==0 and j == 0:
                continue
            try:
                if dict_results['stationary_status']["stat_type"] == "trend":
                    fitted_model = ARIMA(df_series, order=(i , d, j), trend="ct").fit()
                else:
                    fitted_model = ARIMA(df_series, order=(i , d, j)).fit()
                n_ljungBox_results = fitted_model.test_serial_correlation(method="ljungbox")
                n_ljungBox_pValue = n_ljungBox_results[0,1,-1] #gets the p-value of the last lag, portmonteau cumulative test

                # === Save Results ========
                dict_results["models"].append({
                    "order": (i , d , j),
                    "aic": fitted_model.aic,
                    "model": fitted_model,
                    "ljung_box_pValue" : n_ljungBox_pValue  
                })
                # ==========================

                if n_ljungBox_pValue > 0.05: 
                    good_models.append({
                        "order": (i , d , j),
                        "aic": fitted_model.aic,
                        "model": fitted_model 
                    })
                else:
                    print("one lag is not good enough")
            except:
                continue
        if not good_models:
            print("nothing found")
        else:
            best_model = min(good_models, key=lambda x: x["aic"])
            return best_model

def getForecast(ARIMAResults_fitted,fore_length, n_lambda = None):
    #get a clean forecast for the specified length
    #if it has a lambda then use inv boxcox otherwise just forecast

    """
    Generates a forecast with a given length and detransforms the data if necessary.
    This function performs the following operations:
        1. Takes the fitted model and forecasts up to a given length
        2. Applies inverse boxcox if necessary
    Args:
        ARIMAResults_fitted (statsmodels.tsa.arima.model.ARIMAResults): A fitted ARIMA model
        fore_length (int): The length of the forecast
        n_lambda (float): A floating point number to detransform a foregone transformation of the data
    Returns:
        (np.ndarray): An array containing a forecast of fore_length steps
    """

    if n_lambda:

        pred_forecast = ARIMAResults_fitted.forecast(steps=fore_length)
        return inv_boxcox(pred_forecast, n_lambda)

    else:
        return ARIMAResults_fitted.forecast(steps=fore_length)
