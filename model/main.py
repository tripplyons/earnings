from dataset import make_dataset, save_cache, get_recent_features
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, SelectKBest, f_regression
from sklearn.linear_model import LinearRegression, ElasticNetCV
import time


def make_model():
    return make_pipeline(
        StandardScaler(),
        SelectKBest(k=256, score_func=f_regression),
        RFECV(LinearRegression(), step=0.1),
        ElasticNetCV(n_jobs=-1, precompute=False)
    )


def join_predictions(predictions):
    return np.mean(predictions)


def main():
    max_items = 1000
    data_dir = '../scraper/output'
    live_data_dir = '../scraper/live'
    recompute_features = False

    features, labels, tickers, timestamps, market_caps, group_num = make_dataset(
        data_dir,
        max_items=max_items,
        recompute_features=recompute_features
    )
    

    print('training')

    order = np.argsort(timestamps)

    # valid_indices = order[~np.isnan(labels[order])] # doesn't work since it will be out of order
    valid_indices = []
    for i in order:
        if not np.isnan(labels[i]):
            valid_indices.append(i)
    valid_indices = np.array(valid_indices)

    train_rate = 0.9
    num_train_indices = int(len(valid_indices) * train_rate)

    train_indices = np.sort(valid_indices[:num_train_indices])
    test_indices = np.sort(valid_indices[num_train_indices:])

    train_features = features[train_indices]
    train_labels = labels[train_indices]
    test_features = features[test_indices]
    test_labels = labels[test_indices]

    model = make_model()
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
            group_predictions.append(
                join_predictions(model.predict(group_features)))
            group_labels.append(group_label)

    print('Correlation:', np.corrcoef(group_predictions, group_labels)[0, 1])

    save_cache()

    # plt.scatter(group_predictions, group_labels)
    # plt.show()

    start_time = time.time() - 60 * 60 * 24 * 14
    recent_features, recent_tickers, recent_market_caps = get_recent_features(
        live_data_dir, start_time)
    recent_predictions = []

    for i in range(0, len(recent_features)):
        recent_predictions.append(
            join_predictions(model.predict(recent_features[i])))

    z_scores = (recent_predictions - np.mean(recent_predictions)) / \
        np.std(recent_predictions)

    for i in range(0, len(recent_tickers)):
        # if recent_market_caps[i] > 10 ** 8:
        print(recent_tickers[i], z_scores[i])


if __name__ == '__main__':
    main()
