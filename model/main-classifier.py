from dataset import make_dataset, get_recent_features, save_caches
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import time

def make_model():
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])


def train_model(train_features, train_labels):
    model = make_model()

    model.fit(train_features, train_labels)

    return model


def join_predictions(predictions):
    return np.mean(predictions)


def main():
    max_items = 3000
    data_dir = '../scraper/output'
    live_data_dir = '../scraper/live'

    items = make_dataset(
        data_dir,
        max_items=max_items,
        save_every=max_items + 1
    )
    save_caches()

    print('Training...')

    order = np.argsort(items['timestamps'])

    valid_indices = order[np.nonzero(items['labels'][order])]
    valid_indices = valid_indices[items['timestamps'][valid_indices] != 0]
    valid_indices = np.sort(valid_indices)

    train_rate = 0.8
    num_train_indices = int(len(valid_indices) * train_rate)

    # make sure we don't split a group
    while items['group_num'][num_train_indices] == items['group_num'][num_train_indices - 1]:
        num_train_indices -= 1

    train_indices = np.sort(valid_indices[:num_train_indices])
    test_indices = np.sort(valid_indices[num_train_indices:])

    train_features = items['features'][train_indices]
    raw_train_labels = items['labels'][train_indices]
    test_features = items['features'][test_indices]
    raw_test_labels = items['labels'][test_indices]

    train_labels = 1 * (raw_train_labels > 0)
    test_labels = 1 * (raw_test_labels > 0)

    model = train_model(train_features, train_labels)

    test_groups = items['group_num'][test_indices]
    unique_test_groups = np.unique(test_groups)

    group_predictions = []
    group_labels = []
    raw_group_labels = []
    for group in unique_test_groups:
        group_indices = np.arange(len(test_groups))[
            test_groups == group]

        group_features = test_features[group_indices]
        group_label = test_labels[group_indices[0]]
        raw_group_label = raw_test_labels[group_indices[0]]

        valid = ~np.isnan(group_label).any()

        if valid:
            group_predictions.append(
                join_predictions(model.predict_proba(group_features)[:, 1]))
            group_labels.append(group_label)
            raw_group_labels.append(raw_group_label)

    print('Correlation:', np.corrcoef(group_predictions, group_labels)[0, 1])

    group_predictions = np.array(group_predictions)
    group_labels = np.array(group_labels)
    raw_group_labels = np.array(raw_group_labels)

    correct = 1 * (group_predictions > np.mean(group_predictions)) == group_labels

    print('Accuracy:', np.mean(correct), len(correct))

    profit = (group_predictions - np.mean(group_predictions)) * (np.exp(raw_group_labels) - 1)
    print('Mean profit:', np.mean(profit))
    print('Standard deviation:', np.std(profit))

    start_time = time.time() - 60 * 60 * 24 * 2
    recent_features, recent_tickers = get_recent_features(
        live_data_dir, start_time)
    recent_predictions = []

    for i in range(0, len(recent_features)):
        recent_predictions.append(
            join_predictions(model.predict(recent_features[i]))
        )

    recent_predictions = np.array(recent_predictions)

    z_scores = (recent_predictions - np.mean(group_predictions)) / \
        np.std(group_predictions)

    for i in range(0, len(recent_tickers)):
        line = f'{recent_tickers[i]}\t{z_scores[i]:.2f}'
        print(line)


if __name__ == '__main__':
    main()
