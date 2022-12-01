#libraries
import optuna
from optuna import Trial

#download data and analyze
def multi_cripto_predict(symbol, y_coin, time_data, pred, lags, n_obs, y, n_test, 
                         plot_pred=False, save_data=False, save_model=False,
                         n_trials=5000, timeout=7*60*60):

    def objective(trial):
        
        param = {
        #'tree_method':'gpu_hist',  # this parameter means using the GPU when training our model to speedup the training process
        'lambda': trial.suggest_uniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_uniform('alpha', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.01, 1.0),
        'subsample': trial.suggest_uniform('subsample', 0.01, 1.0),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.001, 10.0),
        'n_estimators': 4000,
        'max_depth': trial.suggest_int('max_depth', 1, 100),
        'random_state': trial.suggest_categorical('random_state', [24, 48, 2020]),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
        }

        model = xgb.XGBRegressor(**param)  
        model.fit(df_train, y_train, eval_set=[(df_test, y_test)],
                  early_stopping_rounds=100,
                  verbose=False)
        preds = model.predict(df_test)
        rmse = mean_squared_error(y_test, preds,squared=False)
        return rmse


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

    #RSI
    for lag in range(1, lags):
        rsi(df, y, lag)

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

    #hyperparameter search
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=8)
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    #fit model
    reg = xgb.XGBRegressor(**study.best_trial.params)
    reg.fit(df_train, y_train,
            eval_set=[(df_train, y_train), (df_test, y_test)],
            early_stopping_rounds=100,
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




symbols = ['BTCUSDT',  'ETHUSDT', #'BNBUSDT', 'AXSUSDT',
                       'ETHBTC',  #'BNBBTC',  'AXSBTC', 
                       'ETHEUR'#,  'BNBETH',  'AXSBNB'
          ]

prediccion_ETHEUR, test_ETHEUR, parametros_ETHEUR = multi_cripto_predict(
    symbol = symbols,
    y_coin = 'ETHEUR',
    time_data = '1h',
    pred = 7, 
    lags = 14, 
    n_obs = 365*24, #3000
    y = 'close', 
    n_test = 10, #0.1,
    plot_pred = True,
    save_data = False,
    save_model = True,
    n_trials = 500, 
    timeout = 3*60*60)
