import dataset as ds
from dataset import ClusterDataset
from dataset.util import prepare_time_series, train_valid_test

import numpy as np
from torch.utils.data import Dataset


def load_data_individual(
		_cluster_id,
		_sliding_win,
		cluster_info='../data/cluster_info/cluster_info_a_s.csv'
) -> tuple[Dataset, Dataset, Dataset]:
	df = ds.util.load_clustered_data(
		cluster_info=cluster_info,
		processed_data='../data/processed/moving_norm(5).csv',
	)[_cluster_id]

	cluster_datasets = {'train': [], 'valid': [], 'test': []}
	for series_id in df.columns:
		series_data, num_time_feat = prepare_time_series(df, series_id)
		series_data = series_data.reset_index().rename(columns={'index': 'time_ticks'})

		_train_data, _valid_data, _test_data = train_valid_test(
			series_data,
			train_ratio=0.7,
			test_ratio=0.2,
		)
		cluster_datasets['train'].append(ds.SlidingWinDataset(_train_data, sliding_win=_sliding_win))
		cluster_datasets['valid'].append(ds.SlidingWinDataset(_valid_data, sliding_win=_sliding_win))
		cluster_datasets['test'].append(ds.SlidingWinDataset(_test_data, sliding_win=_sliding_win))

	return (
		ClusterDataset(cluster_datasets['train']),
		ClusterDataset(cluster_datasets['valid']),
		ClusterDataset(cluster_datasets['test']),
	)


def process_dataset_individual(dataset):
	y_array, x_array = [], []
	for i in range(len(dataset)):
		cid, (_y, _x) = dataset[i]
		y_array.append(_y[1].item())
		x_array.append(_x[:, 1].numpy())

	y_array = np.array(y_array)
	x_array = np.array(x_array)
	return x_array, y_array


def train_test(
		model, metrics_fns: list,
		train_X, train_y,
		valid_X, valid_y,
		test_X, test_y,
):
	model.fit(train_X, train_y)

	print("===== Valid result =====")
	valid_y_pred = model.predict(valid_X)
	for metrics_fn in metrics_fns:
		print(f"{metrics_fn.__name__}: {metrics_fn(valid_y, valid_y_pred)}")
	print('========================')

	print("===== Test result =====")
	test_y_pred = model.predict(test_X)
	for metrics_fn in metrics_fns:
		print(f"{metrics_fn.__name__}: {metrics_fn(test_y, test_y_pred)}")
	print('========================')


def individual_model_pipeline(
		cluster_id, sliding_win, cluster_info,
		model, metrics_fns,
):
	print(f'>>>>>>>> cluster {cluster_id} <<<<<<<<')
	train_set, valid_set, test_set = load_data_individual(cluster_id, sliding_win, cluster_info)
	train_x, train_y = process_dataset_individual(train_set)
	valid_x, valid_y = process_dataset_individual(valid_set)
	test_x, test_y = process_dataset_individual(test_set)
	train_test(
		model, metrics_fns,
		train_x, train_y,
		valid_x, valid_y,
		test_x, test_y,
	)
