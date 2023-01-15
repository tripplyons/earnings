import yfinance as yf
from sentence_transformers import SentenceTransformer
import torch
import regex as re
import datetime
import numpy as np
import sys
import hashlib
from caching import Cache


labels_cache = Cache('labels_cache.pkl.gz')
market_caps_cache = Cache('market_caps_cache.pkl.gz')
features_cache = Cache('features_cache.pkl.gz')


def save_caches():
    print('Saving caches...')
    labels_cache.save()
    market_caps_cache.save()
    features_cache.save()


def hash_func(text):
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (10 ** 8)


def get_ticker(text):
    try:
        ticker = re.findall(r'\(.*\:(.*?)\)', text)[0]
        return ticker
    except:
        return None


def get_market_cap(ticker):
    if ticker in market_caps_cache.contents:
        return market_caps_cache.contents[ticker]

    try:
        data = yf.Ticker(ticker)
        cap = data.info['marketCap']
        if cap is None:
            market_caps_cache.contents[ticker] = 0
            return 0
        market_caps_cache.contents[ticker] = cap
        return cap
    except:
        market_caps_cache.contents[ticker] = 0
        return 0


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
    except:
        return 0


def get_label(text, days=10, skip=1):
    try:
        year, month, day = get_timestamp(text)
        ticker = get_ticker(text)

        date_obj = datetime.datetime.strptime(
            f'{month} {day} {year}', '%B %d %Y')
        formatted_date = date_obj.strftime('%Y-%m-%d')

        key = (ticker, formatted_date)
        if key in labels_cache.contents:
            return labels_cache.contents[key]

        data = yf.download(ticker, start=formatted_date, progress=False)
        close = data['Close'].iloc[skip:]
        log_return = np.diff(np.log(close.to_numpy()))
        window = log_return[:days]

        if len(window) != days:
            labels_cache.contents[key] = 0
            return 0

        sharpe = np.sqrt(252) * np.mean(window) / np.std(window)

        if np.isnan(sharpe):
            labels_cache.contents[key] = 0
            return 0
        if not np.isfinite(sharpe):
            labels_cache.contents[key] = 0
            return 0

        labels_cache.contents[key] = sharpe
        return sharpe
    except:
        return 0


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name).to(device)
model_dimension = 384


def get_features(text):
    key = hash_func(text)

    if key in features_cache.contents:
        print('Using cache...')
        return features_cache.contents[key]

    sentences = text.split('\n')

    embeddings = model.encode(sentences)

    features_cache.contents[key] = embeddings
    return embeddings


if __name__ == '__main__':
    text = open(sys.argv[1]).read()
    print(get_label(text))
