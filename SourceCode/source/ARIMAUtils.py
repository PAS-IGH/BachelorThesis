
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
# def getForecast(inverseBoxCoxLmabda fitted model)
def getOptimalModel(df_series_transformed, p, d, q):

    if p > 0 and q == 0:
        return fitAR(df_series_transformed, p, d)["model"]
    elif p == 0 and q > 0:
        return fitMA(df_series_transformed,d, q)["model"]
    else:
        return fitARIMA(df_series_transformed, p, d, q)["model"]


def fitAR(df_series_transformed, p, d):

    p_range = range(max(0, p - 1), p + 2)
    good_models = [] #models whose ljung box pvalue ar not below 0.05
    for i in p_range:
        if i==0:
            continue
        try:
            fitted_model = ARIMA(df_series_transformed, order=(i , d, 0)).fit()
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
    
def fitMA(df_series_transformed,d, q):

    q_range = range(max(0, q - 1), q + 2)
    good_models = [] #models whose ljung box pvalue ar not below 0.05
    for i in q_range:
        if i==0:
            continue
        try:
            fitted_model = ARIMA(df_series_transformed, order=(0 , d , i)).fit()
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
    
def fitARIMA(df_series_transformed,p, d, q):
    p_range = range(max(0, p - 1), p + 2) 
    q_range = range(max(0, q - 1), q + 2)
    good_models = [] #models whose ljung box pvalue ar not below 0.05

    for i in p_range:
        for j in q_range:
            if i==0 and j == 0:
                continue
            try:
                fitted_model = ARIMA(df_series_transformed, order=(i , d , j)).fit()
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