# implement ACF and PACF. The return of the highest function within should be the params for ARIMA later
#check decay rate first, then the significant lags after the first cutoff to diagnose data
# also a figure for plotting is needed or that will be put in front depending on the output 
# of the decay rate

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

# getARIMA_Params (df_data_series) 
# Parameter d gives back d and the
# returns a dictionary with the values for p, d, q

# acfWrapper(df_data_series)
# return UCI_i for all lags, the values for the lags so it can be used for the decay rate

# pacfWrapper(df_data_series)
# return UCI_i for all lags, the values for the lags so it can be used for the decay rate

# Parameter d (bTrend, )
# Get decay rate something which needs to be kept in mind is that ADF/KPSS already gave an indication for 
# differencing/detrending. Therefore implement and test with undifferenced first to see if the decay rate can recognize it
# look at graph of ACF and PACF before and after for pointers
# while decayRate > 0 and decayRate < 0.1 keep differencing and count how often it was differenced that is d, this differenced 
# time series is then acf() to get the UCI for the p and q function

# Parameter p and q ()
# get the minimum cutoff threshold deciding on AR or MA 
# Take the able and just implement the if elif, remember that for case 4 and 5 there is remark 
# on how to choose which model 

# getLagBefCutoff()
# Get parameter M for decay by checking which lag is the last before it dips below the band