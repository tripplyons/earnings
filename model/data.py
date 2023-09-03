from sentence_transformers import SentenceTransformer
import torch
import regex as re
import datetime
import numpy as np
import sys
import hashlib
from caching import Cache
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import os
import pandas as pd

load_dotenv()
api_key = os.getenv('ALPACA_API_KEY')
api_secret = os.getenv('ALPACA_API_SECRET')

labels_cache = Cache('labels_cache.pkl', gz=False)
market_caps_cache = Cache('market_caps_cache.pkl', gz=False)
features_cache = Cache('features_cache.pkl', gz=False)


def save_caches():
    print('Saving caches...')
    labels_cache.save()
    market_caps_cache.save()
    features_cache.save()
    print('Done saving caches.')


def hash_func(text):
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (10 ** 8)


def get_ticker(text):
    try:
        ticker = re.findall(r'\(.*\:(.*?)\)', text)[0]
        return ticker
    except:
        return None


def get_timestamp(text):
    try:
        header = text.split('\n')[0]
        month = header.split(',')[-2].split(' ')[-2]
        day = header.split(',')[-2].split(' ')[-1]
        year = header.split(',')[-1].split(' ')[1]

        return year, month, day
    except:
        return 0, 0, 0


def get_unix_timestamp(year, month, day):
    try:
        timestamp = datetime.datetime.strptime(
            f'{month} {day} {year}', '%B %d %Y').timestamp()
        return timestamp
    except KeyboardInterrupt:
        print('interupted')
        raise
    except:
        return 0


def download_ticker(ticker, start, end):
    timeframe = TimeFrame(1, TimeFrameUnit.Day)
    client = StockHistoricalDataClient(api_key, api_secret)
    request = StockBarsRequest(symbol_or_symbols=ticker, start=start, end=end, timeframe=timeframe)
    bars = client.get_stock_bars(request).df
    bars.index = bars.index.droplevel(0)
    bars.index = pd.to_datetime(bars.stack()).unstack().index
    bars['ticker'] = ticker
    bars['datetime'] = pd.to_datetime(bars.index, unit='s')
    bars['date'] = bars['datetime'].dt.date

    return bars


def get_label(text, days=5, skip=1):
    try:
        year, month, day = get_timestamp(text)
        ticker = get_ticker(text)

        date_obj = datetime.datetime.strptime(
            f'{month} {day} {year}', '%B %d %Y')
        formatted_date = date_obj.strftime('%Y-%m-%d')

        end_date_obj = date_obj + datetime.timedelta(days=2 * days + skip)

        key = (ticker, formatted_date)
        if key in labels_cache.contents:
            return labels_cache.contents[key]

        data = download_ticker(ticker, date_obj, end_date_obj)

        close = data['close'].iloc[skip:]
        log_return = np.diff(np.log(close.to_numpy()))
        window = log_return[:days]

        if len(window) != days:
            labels_cache.contents[key] = 0
            return 0

        total_return = np.sum(window)

        if np.isnan(total_return):
            labels_cache.contents[key] = 0
            return 0
        if not np.isfinite(total_return):
            labels_cache.contents[key] = 0
            return 0

        labels_cache.contents[key] = total_return
        return total_return
    except KeyboardInterrupt:
        print('interrupted')
        raise
    except:
        return 0


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)
model_name = 'thenlper/gte-small'
model = SentenceTransformer(model_name).to(device)
model_dimension = 384
batch_size = 64


def get_features(text):
    key = hash_func(text)

    if key in features_cache.contents:
        print('Using cache...')
        return features_cache.contents[key]

    sentences = text.split('\n')

    batches = [sentences[i:i + batch_size] for i in range(0, len(sentences), batch_size)]
    embeddings = np.zeros((len(sentences), model_dimension))

    for i in range(len(batches)):
        batch = batches[i]
        current_embeddings = model.encode(batch)
        embeddings[len(batch) * i : len(batch) * (i + 1)] = current_embeddings


    features_cache.contents[key] = embeddings
    return embeddings


if __name__ == '__main__':
    text = open(sys.argv[1]).read()
    print(get_label(text))
