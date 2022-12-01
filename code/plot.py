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
