from argparse import ArgumentParser

import pandas as pd

from trainer import (
	RNNs,
	DLinear,
	DLinearPlus,
)


if __name__ == '__main__':
	parser = ArgumentParser(description='Choose Model Experiment')
	parser.add_argument('--model', type=str, default='RTDLinear')

	args = parser.parse_args()

	cluster_info = pd.read_csv('./data/cluster_info/cluster_info_a_s.csv')

	match args.model:
		case 'LSTM': [RNNs.RNNTrainer.train_lstm(cluster_id=cluster_id) for cluster_id in cluster_info['agg clusters'][1:]]
		case 'GRU': [RNNs.RNNTrainer.train_gru(cluster_id=cluster_id) for cluster_id in cluster_info['agg clusters'][1:]]
		case 'DLinear': [DLinear.DLinearTrainer.train_dlinear(cluster_id=cluster_id) for cluster_id in cluster_info['agg clusters']]
		case 'TDLinear': [DLinearPlus.DLinearPlusTrainer.train_td_linear(cluster_id=cluster_id) for cluster_id in cluster_info['agg clusters']]
		case 'RTDLinear': [DLinearPlus.DLinearPlusTrainer.train_rtd_linear(cluster_id=cluster_id) for cluster_id in cluster_info['agg clusters']]
