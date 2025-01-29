import pandas as pd
# from benzinga import financial_data
import requests
# import pandas_ta as ta
import matplotlib.pyplot as plt
from termcolor import colored as cl
import math
# from benzinga import news_data

plt.rcParams['figure.figsize'] = (25,10)
plt.style.use('fivethirtyeight')
# DataFrame tayar karo
data = {'Date': ['2024-01-03', '2024-02-17', '2024-03-29', '2024-04-01'],
        'Close': [1,25,70,100]}

df = pd.DataFrame(data)

# DataFrame ko grafik banao
plt.plot(df['Date'], df['Close'])
plt.title('MSFT Stock Price')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.xticks(rotation=45)
plt.tight_layout()

# Grafik ko PDF banake save karo
plt.savefig('msft_stock_price.pdf')

# Grafik window ko band karo
plt.close()

def get_historical_data(symbol, start_date, interval):
    url = "https://api.benzinga.com/api/v2/bars"
    querystring = {"token": "8766e0baf3cc445498836a7f45aae49e", "symbols": f"{symbol}", "from": f"{start_date}", "interval": f"{interval}"}

    hist_json = requests.get(url, params=querystring).json()
    df = pd.DataFrame(hist_json[0]['candles'])

    return df


aapl = get_historical_data('PSX', '2024-01-01', '1W')
print(aapl)


import requests

url = "https://api.benzinga.com/api/v2/bars"

response = requests.request("GET", url)

print(response.text)


# print(plt)












api_key = "8766e0baf3cc445498836a7f45aae49e"
#
# fin = financial_data.Benzinga(api_key)
#
# stock_rating = fin.ratings()
#
# res =   fin.output(stock_rating)

# print(res)
