#key
coin_marketcap_api_key = '4fd7a1f9-fc43-4c31-8088-a67bbf1e0b26'

#url
url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'

#parameters
parameters = {
  'start':'1',
  'limit':'5000',
  'convert':'USD'
}
headers = {
  'Accepts': 'application/json',
  'X-CMC_PRO_API_KEY': coin_marketcap_api_key,
}

#libraries
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
import datetime
from datetime import date, timedelta
import pandas as pd

#session
session = Session()
session.headers.update(headers)

#get new listed coin
try:
    response = session.get(url, params=parameters)
    data = json.loads(response.text)
    today = date.today()
    yesterday = str(today - timedelta(days=1))
    yesterday_datetime = datetime.datetime.strptime(yesterday, '%Y-%m-%d')
    symbol_list = []
    date_list = []
    for entry in data["data"]:
        symbol = entry["symbol"]
        date_added_str = entry["date_added"][:10]    
        date_added = datetime.datetime.strptime(date_added_str, '%Y-%m-%d')
        if date_added_str >= yesterday:
            symbol_list.append(symbol)
            date_list.append(date_added_str)
        else:
            pass
except (ConnectionError, Timeout, TooManyRedirects) as e:
    print(e)


new_coins = pd.concat([pd.DataFrame(symbol_list, columns=['symbol']), 
                       pd.DataFrame(date_list, columns=['date_added'])],
                      axis=1).sort_values(['date_added', 'symbol'], ascending=[False, True]).reset_index(drop=True)

new_coins.head(10)
