from data import get_label, get_features, get_ticker, get_timestamp, get_market_cap
import os
import numpy as np
import datetime


def get_paths(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                yield os.path.join(root, file)


def get_items(path):
    for file in get_paths(path):
        with open(file) as f:
            text = f.read()
        yield get_features(text), get_label(text), get_ticker(text)


def make_dataset(path, max_items=None):
    labels = []
    features = []
    timestamps = []

    for feature, (timestamp, label), _ in get_items(path):
        if label is not None:
            labels.append(label)
            features.append(feature)
            timestamps.append(timestamp if timestamp is not None else 0)
        if max_items is not None and len(labels) >= max_items:
            break

    labels = np.array(labels)
    features = np.array(features, dtype=object)
    timestamps = np.array(timestamps)

    order = np.argsort(timestamps)

    return features[order], labels[order]


def make_live_dataset(path, max_items=None):
    features = []
    tickers = []
    timestamps = []

    for file in get_paths(path):
        with open(file) as f:
            text = f.read()
        feature = get_features(text)
        ticker = get_ticker(text)
        year, month, day = get_timestamp(text)

        try:
            timestamp = datetime.datetime.strptime(
                f'{month} {day} {year}', '%B %d %Y').timestamp()

            features.append(feature)
            tickers.append(ticker)
            timestamps.append(timestamp if timestamp is not None else 0)
        except ValueError:
            pass

        if max_items is not None and len(features) >= max_items:
            break

    features = np.array(features, dtype=object)
    tickers = np.array(tickers)
    timestamps = np.array(timestamps)

    current_timestamp = datetime.datetime.now().timestamp()
    valid_timestamps = timestamps <= current_timestamp

    features = features[valid_timestamps]
    tickers = tickers[valid_timestamps]
    timestamps = timestamps[valid_timestamps]

    order = np.argsort(timestamps)

    return features[order], tickers[order]
