### API
binance_api_key = '[REDACTED]'    #Enter your own API-key here
binance_api_secret = '[REDACTED]' #Enter your own API-secret here

### CONSTANTS
binsizes = {"1m": 1, "5m": 5, "1h": 60, "1d": 1440}
batch_size = 750
binance_client = Client(api_key=binance_api_key, api_secret=binance_api_secret)


### FUNCTIONS

#binance function
def minutes_of_new_data(symbol, kline_size, data, source):
    if len(data) > 0:  old = parser.parse(data["timestamp"].iloc[-1])
    elif source == "binance": old = datetime.strptime('1 Jan 2017', '%d %b %Y')
    if source == "binance": new = pd.to_datetime(binance_client.get_klines(symbol=symbol, interval=kline_size)[-1][0], unit='ms')
    return old, new

#binance get coin
def get_all_binance(symbol, kline_size, save = False):
    filename = '%s-%s-data.csv' % (symbol, kline_size)
    if os.path.isfile(filename): data_df = pd.read_csv(filename)
    else: data_df = pd.DataFrame()
    oldest_point, newest_point = minutes_of_new_data(symbol, kline_size, data_df, source = "binance")
    delta_min = (newest_point - oldest_point).total_seconds()/60
    available_data = math.ceil(delta_min/binsizes[kline_size])
    if oldest_point == datetime.strptime('1 Jan 2017', '%d %b %Y'): print('Downloading all available %s data for %s. Be patient..!' % (kline_size, symbol))
    else: print('Downloading %d minutes of new data available for %s, i.e. %d instances of %s data.' % (delta_min, symbol, available_data, kline_size))
    klines = binance_client.get_historical_klines(symbol, kline_size, oldest_point.strftime("%d %b %Y %H:%M:%S"), newest_point.strftime("%d %b %Y %H:%M:%S"))
    data = pd.DataFrame(klines, columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore' ])
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='ms')
    if len(data_df) > 0:
        temp_df = pd.DataFrame(data)
        data_df = data_df.append(temp_df)
    else: data_df = data
    data_df.set_index('timestamp', inplace=True)
    if save: data_df.to_csv(filename)
    print('All caught up..!')
    return data_df        

#plot prediction
def plot_fig(df, freq, y_coin):    
    #create variables
    x = df['ds']
    y = df['y']
    yhat = df['yhat']
    df.index = df['ds']
    #define dates for axis
    fig = plt.gcf().autofmt_xdate();
    plt.figure(figsize=(16, 12), dpi=300)
    #define major ticks
    ax = plt.gca()
    xax = ax.get_xaxis()
    if freq == 'H': xax.set_major_locator(dates.DayLocator())
    xax.set_major_formatter(dates.DateFormatter('%d/%b'))
    #define minor ticks
    if freq == 'H': xax.set_minor_locator(dates.HourLocator(byhour=range(0,24,6)))
    xax.set_minor_formatter(dates.DateFormatter('%H:%M'))
    xax.set_tick_params(which='major', pad=15)
    #define plot
    plt.grid(True)
    plt.plot_date(x, y, fmt='-')
    plt.plot_date(x, yhat, fmt='-')
    plt.xlabel('Fecha')
    plt.ylabel("Precio ($)")
    plt.title(f"{y_coin} Price")
    plt.legend(['Observado', 'Predicho'], loc='best')
    #plot
    plt.show();

#optimize types
def optimize_types(dataframe):
    np_types = [np.int8 ,np.int16 ,np.int32, np.int64,
                np.uint8 ,np.uint16, np.uint32, np.uint64]
    np_types = [np_type.__name__ for np_type in np_types]
    type_df = pd.DataFrame(data=np_types, columns=['class_type'])
    type_df['min_value'] = type_df['class_type'].apply(lambda row: np.iinfo(row).min)
    type_df['max_value'] = type_df['class_type'].apply(lambda row: np.iinfo(row).max)
    type_df['range'] = type_df['max_value'] - type_df['min_value']
    type_df.sort_values(by='range', inplace=True)
    for col in dataframe.loc[:, dataframe.dtypes <= np.integer]:
        col_min = dataframe[col].min()
        col_max = dataframe[col].max()
        temp = type_df[(type_df['min_value'] <= col_min) & (type_df['max_value'] >= col_max)]
        optimized_class = temp.loc[temp['range'].idxmin(), 'class_type']
        print("Col name : {} Col min_value : {} Col max_value : {} Optimized Class : {}".format(col, col_min, col_max, optimized_class))
        dataframe[col] = dataframe[col].astype(optimized_class)
    return dataframe

#get results and plot
def get_results(df, freq, y_coin, plot_pred=True):
    #plot figure
    if plot_pred == True: plot_fig(df, freq, y_coin)
    #results
    return  df[df['set'] == 'future'][['yhat']], df[df['set'] == 'test'][['y', 'yhat']].drop_duplicates(subset=['yhat'])

#save model
def save_model(model_name):
    import json
    from fbprophet.serialize import model_to_json, model_from_json
    with open(model_name, 'w') as fout:
        json.dump(model_to_json(m), fout)  # Save model
    with open(model_name, 'r') as fin:
        m = model_from_json(json.load(fin))  # Load model

#feature importance plot
def feat_imp(df, model, n_features):
    d = dict(zip(df.columns, model.feature_importances_))
    ss = sorted(d, key=d.get, reverse=True)
    top_names = ss[0:n_features]
    plt.figure(figsize=(15,15))
    plt.title("Feature importances")
    plt.bar(range(n_features), [d[i] for i in top_names], color="r", align="center")
    plt.xlim(-1, n_features)
    plt.xticks(range(n_features), top_names, rotation='vertical')

#MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
#download data and analyze
def multi_cripto_predict(symbol, y_coin, time_data, pred, lags, n_obs, y, n_test, plot_pred=False, save_data=False,
                         xgb_params={}):

    #validation check
    if lags <= pred: 
        lags = pred+1
        print("\033[1mNone lagged-variable is being used!!!\033[0m\n\n")
        
    #create parameters
    if time_data == '1h': 
        freq = 'H'
        freq_print = 'hourly'
        freq_pred = 'hours'
    if time_data == '1d': 
        freq = 'D'
        freq_print = 'daily'
        freq_pred = 'days'
#     if time_data == '1m': 
#         freq = 'm'
#         freq_print = 'minutely'
#         freq_pred = 'minutes'
#     if time_data == '5m': 
#         freq = 'm'
#         freq_print = 'minutely'
#         freq_pred = 'minutes'

    #information for prints
    target = y_coin + "_" + y
    symbol_print = symbol.copy()
    symbol_print[len(symbol_print)-1] = "and " + symbol_print[len(symbol_print)-1]
    symbol_print = ', '.join([str(elem) for elem in symbol_print]).replace(", and", " and")
    start_time = time.time()
    y_txt=y

    #download data
    print(f"Downloading data from {symbol_print}")
    for coin in symbol[0:1]:
        print(f"Starting with {coin}")
        data = get_all_binance(coin, time_data, save=save_data)
        data = data.add_prefix(str(coin) + "_")
        data.reset_index(inplace=True)
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    for coin in symbol[1:]:
        print(f"Starting with {coin}")
        data_temp = get_all_binance(coin, time_data, save=save_data)
        data_temp = data_temp.add_prefix(str(coin) + "_")
        data_temp.reset_index(inplace=True)
        data_temp['timestamp'] = pd.to_datetime(data_temp['timestamp'])
        data = pd.merge(data, data_temp, how="inner", on=['timestamp'])
    print("\n\nData downloaded...\n\nDataframe optimization:")
    
    #optimize data types
    data = optimize_types(data)
    df = data.reset_index(drop=True)
    del data

    #filter number obs
    df = df.tail(n_obs)

    #create target and time
    df['y'] = df[target].astype(float)
    y = df[target].astype(float); df.drop(target, axis=1, inplace=True)
    df['ds'] = pd.to_datetime(df['timestamp']) + timedelta(hours=2)
    df.drop('timestamp', axis=1, inplace=True)

    # create a future date df
    print(f"\n\nCreating future dataframe...")
    ftr = pd.DataFrame(pd.date_range(df['ds'].max(), periods=pred+1, freq=freq), columns=['ds'])[1:] #1h | 1d
    df = pd.concat([df, ftr], ignore_index=True)        








    ##############################################################################################
    #VER COMO RECORRER TODOS LOS TARGETS PARA SACAR PREDICCION DE TODOS, SIN METER DATOS DE FUTURO
    ##############################################################################################




    #######################################################
    #PENSAR QUE MÁS VARIABLES O INDICADORES SE PUEDEN METER
    #######################################################
    #create lags
    print(f"Doing lags variables...")
    for var in list(set(df.columns) - set(['ds'])):
        df[var] = df[var].astype('float')
        #iterate over lags
        for t in range(pred+1,lags):
            nombre = 'lag'+str(t)+'_'+var
            #create variables
            df[nombre] = df[var].shift(t)
        #borrar origianales
        if var != 'y': df.drop(var, axis=1, inplace=True)

            
            
            
            
    #create moving average  
    for ma in [10, 20, 50, 100, 200]:
        nombre = 'lag'+str(pred+1)+'_ma'+str(ma)
        df[nombre] = df['lag'+str(pred+1)+'_y'].rolling(window=ma).mean()





    #create time variables
    df['date'] = df.index
    df['hour'] = df['ds'].dt.hour
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    df['dayofyear'] = df['ds'].dt.dayofyear
    df['dayofmonth'] = df['ds'].dt.day
    df['weekofyear'] = df['ds'].dt.isocalendar().week.astype(int)
    df.drop('ds', axis=1, inplace=True)

    #create dataframe sizes
    print(f"Splitting dataframes...")
    n_train = int((len(df)-pred-n_test))
    #     n_train = int((len(df)-pred)*(1-test))
    #     n_test = int((len(df)-pred)*(test)) #cambiado para meter un valor entero de observaciones
    n_future = pred

    #split dataframes
    df_train = df.copy().head(n_train).dropna()
    df_test = df.copy()[n_train:n_train+n_test]
    df_future = df.copy().tail(pred)
    y_train = df.head(n_train).dropna()['y']
    y_test = df[n_train:n_train+n_test]['y']
#     y_future = df.tail(pred)['y']

    #drop target from df
    df.drop('y', axis=1, inplace=True)
    df_train.drop('y', axis=1, inplace=True)
    df_test.drop('y', axis=1, inplace=True)
    df_future.drop('y', axis=1, inplace=True)

#     #fit model
#     reg = xgb.XGBRegressor(n_estimators=5000)
#     reg.fit(df_train, y_train,
#             eval_set=[(df_train, y_train), (df_test, y_test)],
#             early_stopping_rounds=50,
#             verbose=False) 
    
    #search hyperparameters
    print("Hyperparameters search...")
    boost = xgb.XGBRegressor()
    regCV = GridSearchCV(boost,
                         xgb_params,
                         cv=5,
                         n_jobs=-1,
                         verbose=0)
    regCV.fit(df_train, y_train,
              eval_set=[(df_train, y_train), (df_test, y_test)],
              early_stopping_rounds = 50,
              verbose=0)

    #fit model
    reg = xgb.XGBRegressor(**regCV.best_params_)
    reg.fit(df_train, y_train,
            eval_set=[(df_train, y_train), (df_test, y_test)],
            early_stopping_rounds=50,
            verbose=0) 
    
    
    
    
    #plot feature importance
    feat_imp(df_train, reg, 20)

    
    
    
    
    #PASAR ESTO A UN DICCIONARIO PARA HACER UN BUCLE FOR EN LUGAR DE 3 BLOQUES
    #make predictions and concatenate
    df_train = pd.concat([df_train.reset_index(drop=True), 
                         pd.DataFrame(y_train).reset_index(drop=True),
                         pd.DataFrame(reg.predict(df_train))], 
                        axis=1).rename(columns={0:'yhat'})
    df_train['ds'] = pd.to_datetime(df_train \
                                    .rename(columns={'dayofmonth':'day'}) \
                                   [['year', 'month', 'day', 'hour']], format = '%Y-%M-%D %H')
    df_test = pd.concat([df_test.reset_index(drop=True), 
                         pd.DataFrame(y_test).reset_index(drop=True),
                         pd.DataFrame(reg.predict(df_test))], 
                        axis=1).rename(columns={0:'yhat'})
    df_test['ds'] = pd.to_datetime(df_test \
                                   .rename(columns={'dayofmonth':'day'}) \
                                   [['year', 'month', 'day', 'hour']], format = '%Y-%M-%D %H')
    df_future = pd.concat([df_future.reset_index(drop=True), 
                         pd.DataFrame(reg.predict(df_future))], 
                        axis=1).rename(columns={0:'yhat'})
    df_future['ds'] = pd.to_datetime(df_future \
                                   .rename(columns={'dayofmonth':'day'}) \
                                   [['year', 'month', 'day', 'hour']], format = '%Y-%M-%D %H')

    
    

    #plot prediction
    if plot_pred == True: plot_fig(df_test, '1h', y_coin)

    #print info
    mape = mean_absolute_percentage_error(y_true=df_test['y'], y_pred=df_test['yhat'])
    print(f"\n\n\033[1mDone!\033[0m\n\n"
          f"Execution time: \033[1m{round((time.time() - start_time)/60, 1)} minutes \033[0m\n" 
          f"Used \033[1m{str(y_coin)}\033[0m (\033[1m{str(y_txt)}\033[0m price) as target\n" 
          f"Info from \033[1m{symbol_print}\033[0m prices\n"
          f"Analyzed dates between \033[1m{df_train.ds.min()}\033[0m and \033[1m{df_test.ds.max()}\033[0m\n"
          f"Data in \033[1m{freq_print}\033[0m aggregation\n"
          f"\033[1mPredicted {pred} {freq_pred}\033[0m for future data\n"
          f"Variables time in \033[1m{lags} lags\033[0m were used for " 
          f"\033[1m{df.shape[0]} observations\033[0m, " 
          f"resulting in \033[1m{df.shape[1]} columns\033[0m\n\n"
          f"\033[1mMean Absoluto Percentaje Error = {round(mape, 2)}%\033[0m\n\n"
          f"\033[1mHyperparameters for XGBoost:\033[0m\n{json.dumps(reg.get_params(),indent=4,sort_keys=True)}\n\n")

    #get results
    if save_model == True: 
        save_model(y_coin+".json")
        print("Model saved...")
    if save_data == True: 
        df.to_csv(y_coin+y+"_"+time_date+"_"+str(pred)+"pred_"+str(lags)+".csv", index=False)
        print("Data saved...")
    return df_future, df_test, regCV.best_params_

#validation
def validation(y_coin, y, df_res, plot_pred=False):

    #define target 
    symbol = y_coin

    #check periodity of data
    df_res = df_res.copy()
    df_res.set_index('ds', drop=True, inplace=True)
    if df_res.index[0] + timedelta(minutes=1) == df_res.index[1]: time_data = '1m'
    if df_res.index[0] + timedelta(hours=1) == df_res.index[1]: time_data = '1h'
    if df_res.index[0] + timedelta(days=1) == df_res.index[1]: time_data = '1d'

    #define freqs
    if time_data == '1h': 
        freq = 'H'
        freq_print = 'hourly'
        freq_pred = 'hours'
    if time_data == '1d': 
        freq = 'D'
        freq_print = 'daily'
        freq_pred = 'days'

    #download data
    data_temp = get_all_binance(y_coin, time_data, save=False)
    data_temp = data_temp[[y]].rename(columns={y: 'y'})
    data_temp.index.names = ['ds']
    data_temp['y'] = data_temp['y'].astype(float)
    data_temp.index = pd.to_datetime(data_temp.index)
    data_temp.index = data_temp.index + timedelta(hours = 2)
    data_temp.reset_index(inplace=True)

    #merge dataframes
    data = pd.merge(df_res, data_temp, how="inner", on=['ds'])
    data = data[['ds', 'y', 'yhat'] + list(set(data.columns)-set(['ds', 'y', 'yhat']))]

    
    #warning or results
    if data.shape[0] == 0: print(f"\nThere is not overlapping dates...\n"
                                 f"....... Actual date is from {data_temp['ds'].min()} to {data_temp['ds'].max()}\n"
                                 f"... Prediction date is from {df_res.reset_index()['ds'].min()} to {df_res.reset_index()['ds'].max()}")
    elif data.shape[0] > 0:
        #plot
        if plot_pred == True: plot_fig(df_res=data, freq=freq, y_coin=y_coin)
        #results
        return data
