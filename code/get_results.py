#get results and plot
def get_results(df, freq, y_coin, plot_pred=True):
    #plot figure
    if plot_pred == True: plot_fig(df, freq, y_coin)
    #results
    return  df[df['set'] == 'future'][['yhat']], df[df['set'] == 'test'][['y', 'yhat']].drop_duplicates(subset=['yhat'])
