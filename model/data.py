import yfinance as yf
from sentence_transformers import SentenceTransformer
import torch
import regex as re
import datetime
import numpy as np
import sys
import pickle
import hashlib


def hash_func(text):
    return int(hashlib.sha256(text.encode('utf-8')).hexdigest(), 16) % (10 ** 8)


def load_cache():
    try:
        with open('cache.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return {
            'labels': {},
            'features': {}
        }


cache = load_cache()


def clear_features():
    global cache
    cache['features'] = {}
    save_cache()


def save_cache():
    global cache
    with open('cache.pkl', 'wb') as f:
        pickle.dump(cache, f)


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


def get_label(text, days=10, skip=1):
    global cache

    if hash_func(text) in cache['labels']:
        return cache['labels'][hash_func(text)]

    try:
        year, month, day = get_timestamp(text)
        ticker = get_ticker(text)

        date_obj = datetime.datetime.strptime(
            f'{month} {day} {year}', '%B %d %Y')
        formatted_date = date_obj.strftime('%Y-%m-%d')
        unix_time = int(date_obj.timestamp())

        data = yf.download(ticker, start=formatted_date)
        close = data['Close'].iloc[skip:]
        log_return = np.diff(np.log(close.to_numpy()))
        window = log_return[:days]

        if len(window) != days:
            print('Not enough data for', ticker, formatted_date)
            cache['labels'][hash_func(text)] = (None, None)
            return None, None

        # sharpe = np.sqrt(252) * np.mean(window) / np.std(window)
        sharpe = np.mean(window)

        if np.isnan(sharpe):
            print('Nan for', ticker, formatted_date)
            cache['labels'][hash_func(text)] = (None, None)
            return None, None
        if not np.isfinite(sharpe):
            print('Not finite for', ticker, formatted_date)
            cache['labels'][hash_func(text)] = (None, None)
            return None, None

        cache['labels'][hash_func(text)] = (unix_time, sharpe)
        return unix_time, sharpe
    except:
        cache['labels'][hash_func(text)] = (None, None)
        return None, None


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_name = 'sentence-transformers/all-mpnet-base-v2'
# model_name = 'sentence-transformers/sentence-t5-base'
# model_name = 'sentence-transformers/all-MiniLM-L6-v2'
model = SentenceTransformer(model_name).to(device)


def get_features(text):
    global cache

    print(len(cache['features']))

    if hash_func(text) in cache['features']:
        return cache['features'][hash_func(text)]

    # api_text = text.replace('\n', ' ').replace('\t', ' ')
    # embedding = openai.Embedding.create(
    #     input=[api_text], model='text-similarity-babbage-001'
    # )[0]
    # sentences = text.split('\n')
    # embedding = model.encode(sentences).mean(axis=0)
    # embedding = model.encode(text)
    # cache['features'][hash_func(text)] = embedding

    # max_chars = 1000
    sentences = text.split('\n')
    # groups = []
    # buffer = ''
    # for i in range(0, len(sentences)):
    #     if len(buffer) + len(sentences[i]) > max_chars:
    #         groups.append(buffer)
    #         buffer = ''
    #     buffer += sentences[i] + '\n'
    # groups.append(buffer)

    embeddings = []
    # for group in groups:
    #     api_text = group.replace('\n', ' ').replace('\t', ' ')

    #     # response = openai.Embedding.create(
    #     #     input=api_text,
    #     #     model="text-similarity-babbage-001"
    #     # )
    #     # embedding = np.array(response['data'][0]['embedding'])

    #     embedding = model.encode(api_text)

    #     embeddings.append(embedding)

    embeddings = model.encode(sentences)

    # embedding = np.array(embeddings).mean(axis=0)
    # embeddings = np.array([embedding])

    # embeddings = np.array([model.encode(text)])

    cache['features'][hash_func(text)] = embeddings

    if (len(cache['features']) % 100 == 0):
        save_cache()

    # print(embedding)
    return embeddings


if __name__ == '__main__':
    text = open(sys.argv[1]).read()
    print(get_label(text))
