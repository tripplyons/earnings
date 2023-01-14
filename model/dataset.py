from data import get_label, get_features, get_ticker, get_timestamp, get_market_cap, get_unix_timestamp, model_dimension
import os
import numpy as np
import h5py

cache = h5py.File('cache.hdf5', 'a')

def get_paths(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                yield os.path.join(root, file)


def get_items(path):
    for file in get_paths(path):
        with open(file) as f:
            text = f.read()
        yield text


def save_cache():
    cache.close()


def make_dataset(path, max_items=None, recompute_features=False):
    # do everything through the cache to avoid reprocessing
    # the same data over and over again
    features = cache.get('features')
    labels = cache.get('labels')
    tickers = cache.get('tickers')
    timestamps = cache.get('timestamps')
    market_caps = cache.get('market_caps')
    group_num = cache.get('group_num')
    current_group_num = 0
    current_index = 0

    try:
        if features is None:
            features = cache.create_dataset(
                'features', (0, model_dimension), maxshape=(None, None))
            labels = cache.create_dataset('labels', (0,), maxshape=(None,))
            tickers = cache.create_dataset(
                'tickers', (0,), maxshape=(None,), dtype=h5py.string_dtype(encoding='utf-8'))
            timestamps = cache.create_dataset(
                'timestamps', (0,), maxshape=(None,))
            market_caps = cache.create_dataset(
                'market_caps', (0,), maxshape=(None,))
            group_num = cache.create_dataset(
                'group_num', (0,), maxshape=(None,))
        for i, item in enumerate(get_items(path)):
            if max_items is not None and i >= max_items:
                break
            if i % 10 == 0:
                print(i)

            ticker = get_ticker(item)
            ticker = ticker if ticker is not None else ''
            timestamp = get_timestamp(item)
            unix_timestamp = get_unix_timestamp(
                timestamp[0], timestamp[1], timestamp[2])

            matching_timestamps = np.equal(timestamps, unix_timestamp)
            matching_tickers = tickers[matching_timestamps]
            found_match = False
            for matching_ticker in matching_tickers:
                if matching_ticker == bytes(ticker, 'utf-8'):
                    found_match = True
                    break

            if found_match:
                print('skipping', ticker, timestamp)

                if recompute_features:
                    current_features = get_features(item)
                    for feature in current_features:
                        features[current_index] = feature
                        current_index += 1

                continue

            current_features = get_features(item)
            label = get_label(item)
            market_cap = get_market_cap(item)

            for feature in current_features:
                features.resize(features.shape[0] + 1, axis=0)
                features[-1] = feature
                labels.resize(labels.shape[0] + 1, axis=0)
                labels[-1] = label
                tickers.resize(tickers.shape[0] + 1, axis=0)
                tickers[-1] = ticker
                timestamps.resize(timestamps.shape[0] + 1, axis=0)
                timestamps[-1] = unix_timestamp
                market_caps.resize(market_caps.shape[0] + 1, axis=0)
                market_caps[-1] = market_cap
                group_num.resize(group_num.shape[0] + 1, axis=0)
                group_num[-1] = current_group_num

                current_index += 1

            current_group_num += 1
    except KeyboardInterrupt:
        print('interrupted')
        save_cache()
        raise

    return features, labels, tickers, timestamps, market_caps, group_num


def get_recent_features(path, start_time):
    features = []
    tickers = []
    timestamps = []
    market_caps = []

    for i, item in enumerate(get_items(path)):
        ticker = get_ticker(item)
        ticker = ticker if ticker is not None else ''
        timestamp = get_timestamp(item)
        unix_timestamp = get_unix_timestamp(
            timestamp[0], timestamp[1], timestamp[2])

        if unix_timestamp is not None and unix_timestamp >= start_time:
            features.append(get_features(item))
            tickers.append(ticker)
            timestamps.append(unix_timestamp)
            market_caps.append(get_market_cap(ticker))

    order = np.argsort(timestamps)

    ordered_features = [features[i] for i in order]
    ordered_tickers = [tickers[i] for i in order]
    ordered_market_caps = [market_caps[i] for i in order]

    return ordered_features, ordered_tickers, ordered_market_caps
