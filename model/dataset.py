from data import get_label, get_features, get_ticker, get_timestamp, get_unix_timestamp, save_caches
import os
import numpy as np

def get_paths(path):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.txt'):
                yield os.path.join(root, file)


def get_items(path):
    i = 0
    for file in get_paths(path):
        try:
            with open(file) as f:
                text = f.read()
            yield text
        except:
            print('error reading', file, i)
        
        i += 1


def make_dataset(path, max_items=None, save_every=100):
    items = {
        'timestamps': [],
        'tickers': [],
        'features': [],
        'labels': [],
        'group_num': []
    }
    current_group_num = 0
    current_index = 0

    try:
        for i, item in enumerate(get_items(path)):
            print(f'{i} out of {max_items}')

            if max_items is not None and i >= max_items:
                break
            if i % save_every == 0 and i != 0:
                print(i)
                save_caches()

            ticker = get_ticker(item)
            ticker = ticker if ticker is not None else ''
            timestamp = get_timestamp(item)
            unix_timestamp = get_unix_timestamp(
                timestamp[0], timestamp[1], timestamp[2])

            matching_timestamps = np.equal(items['timestamps'], unix_timestamp)
            matching_tickers = [items['tickers'][i] for i in np.where(matching_timestamps)[0]]
            found_match = False
            for matching_ticker in matching_tickers:
                if matching_ticker == bytes(ticker, 'utf-8'):
                    found_match = True
                    break

            if found_match:
                print('skipping', ticker, timestamp)
                continue

            current_features = get_features(item)
            label = get_label(item)

            for feature in current_features:
                items['features'].append(feature)
                items['labels'].append(label)
                items['tickers'].append(ticker)
                items['timestamps'].append(unix_timestamp)
                items['group_num'].append(current_group_num)

                current_index += 1

            current_group_num += 1
    except KeyboardInterrupt:
        print('interrupted')
        raise

    items_np = {}
    for key in items:
        items_np[key] = np.array(items[key])
    
    return items_np


def get_recent_features(path, start_time):
    features = []
    tickers = []
    timestamps = []

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

    order = np.argsort(timestamps)

    ordered_features = [features[i] for i in order]
    ordered_tickers = [tickers[i] for i in order]

    return ordered_features, ordered_tickers
