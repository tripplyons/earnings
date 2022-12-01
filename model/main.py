from dataset import make_dataset, make_live_dataset
from data import save_cache, get_ticker, get_market_cap
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import resource


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 // 2, hard))


def match_features(features, labels):
    matched_features = []
    matched_labels = []

    for i in range(0, len(features)):
        for feature in features[i]:
            matched_features.append(feature)
            matched_labels.append(labels[i])

    return matched_features, matched_labels


def main():
    memory_limit()

    max_items = None
    data_dir = '../scraper/output'
    features, labels = make_dataset(data_dir, max_items=max_items)
    save_cache()
    # np.savetxt('features.txt', features)
    # np.savetxt('labels.txt', labels)
    # features = np.loadtxt('features.txt')
    # labels = np.loadtxt('labels.txt')

    valid_indices = np.arange(len(labels))

    train_rate = 0.9
    num_train = int(len(valid_indices) * train_rate)

    train_features = features[valid_indices[:num_train]]
    train_labels = labels[valid_indices[:num_train]]
    test_features = features[valid_indices[num_train:]]
    test_labels = labels[valid_indices[num_train:]]

    matched_train_features, matched_train_labels = match_features(
        train_features, train_labels
    )
    matched_test_features, matched_test_labels = match_features(
        test_features, test_labels
    )

    model = make_pipeline(
        StandardScaler(),
        # PCA(n_components=0.95),
        # StandardScaler(),
        LinearRegression()
        # RandomForestRegressor(random_state=0)
        # GradientBoostingRegressor(n_estimators=10, max_depth=6)
        # HistGradientBoostingRegressor(max_iter=1000)
    )
    model.fit(matched_train_features, matched_train_labels)

    test_preds = []

    for i in range(0, len(test_features)):
        pred = np.mean(model.predict(test_features[i]))
        test_preds.append(pred)

    test_preds = np.array(test_preds)

    test_correlation = np.corrcoef(test_preds, test_labels)[0, 1]

    print('Test correlation:', test_correlation)

    matched_test_preds = model.predict(matched_test_features)

    num_random_preds = 10000
    random_preds = np.random.randn(len(matched_test_preds) * num_random_preds).reshape(
        num_random_preds, len(matched_test_preds)
    )
    random_correlations = []
    for random_pred in random_preds:
        random_correlations.append(np.corrcoef(
            random_pred, matched_test_labels)[0, 1])
    random_correlations = np.array(random_correlations)

    print("Test correlation p-value:",
          np.mean(test_correlation < random_correlations))

    print('Matched test correlation:', np.corrcoef(
        matched_test_preds, matched_test_labels)[0, 1])

    all_features, all_tickers = make_live_dataset(
        data_dir, max_items=max_items)
    num_latest = 100
    chosen_features = all_features[-num_latest:]
    chosen_tickers = all_tickers[-num_latest:]
    chosen_market_caps = list(map(get_market_cap, chosen_tickers))

    train_preds = []

    for i in range(0, len(train_features)):
        pred = np.mean(model.predict(train_features[i]))
        train_preds.append(pred)

    train_preds = np.array(train_preds)

    for i in range(0, len(chosen_features)):
        pred = np.mean(model.predict(chosen_features[i]))
        z_pred = (pred - np.mean(train_preds)) / np.std(train_preds)

        if chosen_market_caps[i] > 10 ** 8:
            # print(chosen_tickers[i], pred, z_pred)
            print(chosen_tickers[i], z_pred)

    # plt.scatter(test_preds, test_labels)
    # plt.show()


if __name__ == '__main__':
    main()
