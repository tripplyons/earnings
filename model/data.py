import yfinance as yf
from sentence_transformers import SentenceTransformer
import torch
import regex as re
import datetime
import numpy as np
import sys
import hashlib


def hash_func(text):
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (10 ** 8)


def get_ticker(text):
    try:
        ticker = re.findall(r'\(.*\:(.*?)\)', text)[0]
        return ticker
    except:
        return None


def get_market_cap(ticker):
    try:
        data = yf.Ticker(ticker)
        cap = data.info['marketCap']
        if cap is None:
            return 0
        return cap
    except:
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
        return None


def get_label(text, days=10, skip=1):
    try:
        year, month, day = get_timestamp(text)
        ticker = get_ticker(text)

        date_obj = datetime.datetime.strptime(
            f'{month} {day} {year}', '%B %d %Y')
        formatted_date = date_obj.strftime('%Y-%m-%d')

        data = yf.download(ticker, start=formatted_date, progress=False)
        close = data['Close'].iloc[skip:]
        log_return = np.diff(np.log(close.to_numpy()))
        window = log_return[:days]

        if len(window) != days:
            return None

        sharpe = np.sqrt(252) * np.mean(window) / np.std(window)

        if np.isnan(sharpe):
            return None
        if not np.isfinite(sharpe):
            return None

        return sharpe
    except:
        return None


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_name = 'sentence-transformers/all-mpnet-base-v2'
model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name).to(device)


def get_features(text):
    sentences = text.split('\n')

    embeddings = model.encode(sentences)

    return embeddings


if __name__ == '__main__':
    text = open(sys.argv[1]).read()
    print(get_label(text))
