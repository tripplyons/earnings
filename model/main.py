from dataset import make_dataset, get_recent_features
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVR
from data import save_caches
import time
from models import MetaRegressor

def make_model():
    return make_pipeline(
        StandardScaler(),
        Nystroem(n_components=1000, n_jobs=-1),
        LinearSVR()
    )

def make_meta_model(make_model):
    return MetaRegressor(
        make_model(),
        make_model()
    )

def train_model(train_features, train_labels):
    model = make_meta_model(make_model)

    model.fit(train_features, train_labels)

    return model


def join_predictions(predictions):
    return np.mean(predictions)


def main():
    max_items = 2000
    data_dir = '../scraper/output'
    live_data_dir = '../scraper/live'

    items = make_dataset(
        data_dir,
        max_items=max_items,
        save_every=1000
    )
    
    save_caches()
    
    print('Training...')

    order = np.argsort(items['timestamps'])

    valid_indices = order[np.nonzero(items['labels'][order])]
    valid_indices = valid_indices[items['timestamps'][valid_indices] != 0]
    valid_indices = np.sort(valid_indices)

    train_rate = 0.9
    num_train_indices = int(len(valid_indices) * train_rate)

    # make sure we don't split a group
    while items['group_num'][num_train_indices] == items['group_num'][num_train_indices - 1]:
        num_train_indices -= 1

    train_indices = np.sort(valid_indices[:num_train_indices])
    test_indices = np.sort(valid_indices[num_train_indices:])

    train_features = items['features'][train_indices]
    train_labels = items['labels'][train_indices]
    test_features = items['features'][test_indices]
    test_labels = items['labels'][test_indices]

    model = train_model(train_features, train_labels)

    test_groups = items['group_num'][test_indices]
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

    start_time = time.time() - 60 * 60 * 24 * 3
    recent_features, recent_tickers, recent_market_caps = get_recent_features(
        live_data_dir, start_time)
    recent_predictions = []

    for i in range(0, len(recent_features)):
        recent_predictions.append([
            join_predictions(model.regressor.predict(recent_features[i])),
            join_predictions(model.predict(recent_features[i]))
        ])

    recent_predictions = np.array(recent_predictions)
    
    z_scores = recent_predictions
    z_scores[:, 0] -= np.mean(recent_predictions[:, 0])
    z_scores[:, 0] /= np.std(recent_predictions[:, 0])
    z_scores[:, 1] -= np.mean(recent_predictions[:, 1])
    z_scores[:, 1] /= np.std(recent_predictions[:, 1])

    for i in range(0, len(recent_tickers)):
        line = f'{recent_tickers[i]}\t{z_scores[i][0]:.2f}\t{z_scores[i][1]:.2f}'
        print(line)


if __name__ == '__main__':
    main()
