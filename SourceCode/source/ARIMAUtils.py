
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from coreforecast.scalers import inv_boxcox
# def getForecast(inverseBoxCoxLmabda fitted model)
def getOptimalModel(df_series, p, d, q):
    """
    Gets the optimal model based on given params and ljung box test
    This method performs the following operations:
        1. Based on paramater decides which model to fit via a simple grid search
    Args:
        df_series(pandas.DataSeries): The data upon which the model is supposed to be fitted
        p (int): An integer dictating the AR part of ARIMA
        d (int): An integer dictating the differencing part of ARIMA
        q (int): An integer dictating the MA part of ARIMA
    Returns:
        An optimally fitted model
    """
    if p > 0 and q == 0:
        return fitAR(df_series, p, d)["model"]
    elif p == 0 and q > 0:
        return fitMA(df_series,d, q)["model"]
    else:
        return fitARIMA(df_series, p, d, q)["model"]


def fitAR(df_series, p, d):
    """
    Gets the optimal model based on given AR params and ljung box test.
    This method performs the following operations:
        1. Based on paramater builds the appropriate +-1 range and loops through that range
        2. Checks with ARIMAResult.test_serial_correlation(method="ljungbox") if the p-value is freater than 0.05
        3. If that condition is fullfilled add to a list a dictionary with the order, aic value and model
        4. Check the AIC values in the model list with one another and take the item with the lowest AIC 
    Args:
        df_series(pandas.DataSeries): The data upon which the model is supposed to be fitted
        p (int): An integer dictating the AR part of ARIMA
        d (int): An integer dictating the differencing part of ARIMA
    Returns:
        An optimally fitted AR model
    """
    p_range = range(max(0, p - 1), p + 2)
    good_models = [] #models whose ljung box pvalue ar not below 0.05
    for i in p_range:
        if i==0:
            continue
        try:
            fitted_model = ARIMA(df_series, order=(i , d, 0)).fit()
            n_ljungBox_results = fitted_model.test_serial_correlation(method="ljungbox")
            n_ljungBox_pValue = n_ljungBox_results[0,1,-1] #gets the p-value of the last lag, portmonteau cumulative test

            if n_ljungBox_pValue > 0.05: 
                good_models.append({
                    "order": (i , d, 0),
                    "aic": fitted_model.aic,
                    "model": fitted_model 
                })
            else:
                print("one lag is not good enough")
        except:
            continue
    if not good_models:
        print("nothing found AR")
    else:
        best_model = min(good_models, key=lambda x: x["aic"])
        return best_model
    
def fitMA(df_series,d, q):

    """
    Gets the MA optimal model based on given MA params and ljung box test.
    This method performs the following operations:
        1. Based on paramater builds the appropriate +-1 range and loops through that range
        2. Checks with ARIMAResult.test_serial_correlation(method="ljungbox") if the p-value is freater than 0.05
        3. If that condition is fullfilled add to a list a dictionary with the order, aic value and model
        4. Check the AIC values in the model list with one another and take the item with the lowest AIC 
    Args:
        df_series(pandas.DataSeries): The data upon which the model is supposed to be fitted
        d (int): An integer dictating the differencing part of ARIMA
        q (int): An integer dictating the MA part of ARIMA
    Returns:
        An optimally fitted MA model
    """
    q_range = range(max(0, q - 1), q + 2)
    good_models = [] #models whose ljung box pvalue ar not below 0.05
    for i in q_range:
        if i==0:
            continue
        try:
            fitted_model = ARIMA(df_series, order=(0 , d , i)).fit()
            n_ljungBox_results = fitted_model.test_serial_correlation(method="ljungbox")
            n_ljungBox_pValue = n_ljungBox_results[0,1,-1] #gets the p-value of the last lag, portmonteau cumulative test

            if n_ljungBox_pValue > 0.05: 
                good_models.append({
                    "order": (0 , d , i),
                    "aic": fitted_model.aic,
                    "model": fitted_model 
                })
            else:
                print("one lag is not good enough")
        except Exception as e:
            print(e)
            continue
    if not good_models:
        print("nothing found MA")
    else:
        best_model = min(good_models, key=lambda x: x["aic"])
        return best_model
    
def fitARIMA(df_series,p, d, q):

    """
    Gets the optimal model based on given ARMA params and ljung box test.
    This method performs the following operations:
        1. Based on paramaters builds the appropriate +-1 range and loops through that range
        2. Checks with ARIMAResult.test_serial_correlation(method="ljungbox") if the p-value is freater than 0.05
        3. If that condition is fullfilled add to a list a dictionary with the order, aic value and model
        4. Check the AIC values in the model list with one another and take the item with the lowest AIC 
    Args:
        df_series(pandas.DataSeries): The data upon which the model is supposed to be fitted
        p (int): An integer dictating the AR part of ARIMA
        d (int): An integer dictating the differencing part of ARIMA
        q (int): An integer dictating the MA part of ARIMA
    Returns:
        An optimally fitted ARMA model
    """
    p_range = range(max(0, p - 1), p + 2) 
    q_range = range(max(0, q - 1), q + 2)
    good_models = [] #models whose ljung box pvalue ar not below 0.05

    for i in p_range:
        for j in q_range:
            if i==0 and j == 0:
                continue
            try:
                fitted_model = ARIMA(df_series, order=(i , d , j)).fit()
                n_ljungBox_results = fitted_model.test_serial_correlation(method="ljungbox")
                n_ljungBox_pValue = n_ljungBox_results[0,1,-1] #gets the p-value of the last lag, portmonteau cumulative test

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
    Generates a forecast with a given length and detransforms the data if necessary
    This method performs the following operations:
        1. Takes the fitted model and forecasts up to a given length
        2. Applies inverse boxcox if necessary

    Args:
        ARIMAResults_fitted (statsmodels.tsa.arima.model.ARIMAResults): A fitted ARIMA model
        fore_length(int): The length of the forecast
        n_lambda(float): A floating point to detransform a foregone transformation of the data
    Returns:
        An array containing the forecast for fore_length steps

    """

    if n_lambda:

        pred_forecast = ARIMAResults_fitted.forecast(steps=fore_length)
        return inv_boxcox(pred_forecast, n_lambda)
    else:
        return ARIMAResults_fitted.forecast(steps=fore_length)