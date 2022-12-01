from dataset import make_dataset, save_cache, get_recent_features
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, BaggingRegressor
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import time


def main():
    max_items = 2000
    data_dir = '../scraper/output'
    live_data_dir = '../scraper/live'
    features, labels, tickers, timestamps, market_caps, group_num = make_dataset(
        data_dir, max_items=max_items)
    print('training')

    order = np.argsort(timestamps)

    # valid_indices = order[~np.isnan(labels[order])]
    valid_indices = []
    for i in order:
        if not np.isnan(labels[i]):
            valid_indices.append(i)
    valid_indices = np.array(valid_indices)

    train_rate = 0.8
    num_train_indices = int(len(valid_indices) * train_rate)

    train_indices = np.sort(valid_indices[:num_train_indices])
    test_indices = np.sort(valid_indices[num_train_indices:])

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]

    model = make_pipeline(
        StandardScaler(),
        # PCA(n_components=0.95),
        # StandardScaler(),
        # LinearRegression()
        # RandomForestRegressor(random_state=0)
        # GradientBoostingRegressor(n_estimators=10, max_depth=6)
        # HistGradientBoostingRegressor()
        BaggingRegressor(
            base_estimator=HistGradientBoostingRegressor(
                max_iter=10,
                learning_rate=1.0
            ),
            n_estimators=10,
        )
    )
    model.fit(train_features, train_labels)

    test_groups = group_num[test_indices]
    unique_test_groups = np.unique(test_groups)

    group_predictions = []
    group_labels = []
    for group in unique_test_groups:
        group_indices = np.arange(len(test_groups))[
            test_groups == group]

        group_features = test_features[group_indices]
        group_label = test_labels[group_indices[0]]

        valid = ~np.isnan(group_label).any()

        if valid:
            group_predictions.append(np.mean(model.predict(group_features)))
            group_labels.append(group_label)

    print('Correlation:', np.corrcoef(group_predictions, group_labels)[0, 1])

    save_cache()

    # plt.scatter(group_predictions, group_labels)
    # plt.show()

    start_time = time.time() - 60 * 60 * 24 * 1
    recent_features, recent_tickers, recent_market_caps = get_recent_features(
        live_data_dir, start_time)
    recent_predictions = []

    for i in range(0, len(recent_features)):
        recent_predictions.append(
            np.mean(model.predict(recent_features[i])))

    z_scores = (recent_predictions - np.mean(recent_predictions)) / \
        np.std(recent_predictions)

    for i in range(0, len(recent_tickers)):
        if recent_market_caps[i] > 10 ** 8:
            print(recent_tickers[i], z_scores[i])


if __name__ == '__main__':
    main()
