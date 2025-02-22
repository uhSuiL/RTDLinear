from numpy.fft import rfft
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import numpy as np
import pandas as pd


def z_score(df: pd.DataFrame, col_name: str):
	_col = df[col_name]
	_mean = _col.mean()
	_std = _col.std()
	df[col_name] = (_col - _mean) / _std
	return df


def moving_norm(
		data: pd.DataFrame | pd.Series,
		win_len: int = 20,
		drop_na: bool = False,
		anomaly_pos: list = None,
		r = 0.2
):
	data_norm = data.copy() * pd.NA
	avg_df, std_df = [], []
	for i in range(0, data.shape[0], win_len):
		df = data.iloc[i: i+win_len]
		avg, std = df.mean(), df.std()
		data_norm.iloc[i: i+win_len] = (df - avg) / std

		avg_df.append(avg)
		std_df.append(std)

		if anomaly_pos is not None:
			anomaly_pos += df.stack()[(abs(df - avg) > r * std).stack()].index.tolist()

	avg_df, std_df = pd.DataFrame(avg_df), pd.DataFrame(std_df)
	assert avg_df.shape == std_df.shape, (avg_df.shape, std_df.shape)

	if drop_na:
		ill_cols = data_norm.columns[data_norm.isna().any()]
		data_norm = data_norm.drop(ill_cols, axis=1)
		avg_df = avg_df.drop(ill_cols, axis=1)
		std_df = std_df.drop(ill_cols, axis=1)

	# assert pd.isna(data_norm).to_numpy().sum() == 0
	return data_norm, avg_df, std_df


def softmax(data: np.array, axis=0) -> np.array:
	exp_data = np.exp(data)
	return exp_data / exp_data.sum(axis=axis)


def gen_features(data: pd.DataFrame, power = 1/2.5, avg_scale = 1/1e5, std_scale = 1/1e5, *, concat = True):
	data_norm, avg_df, std_df = moving_norm(data, drop_na=True)

	fft_result = rfft(data_norm, axis=0)
	freq_feat = softmax(abs(fft_result.imag) ** power)  # (T/2, num_series)
	phase_feat = softmax(abs(fft_result.real) ** power)  # (T/2, num_series)

	avg_feat = softmax((avg_df * avg_scale).to_numpy())  # (d, num_series)
	std_feat = softmax((std_df * std_scale).to_numpy())  # (d, num_series)

	assert freq_feat.shape[-1] == phase_feat.shape[-1] == avg_feat.shape[-1] == std_feat.shape[-1], \
		(freq_feat.shape, phase_feat.shape, avg_feat.shape, std_feat.shape)

	if concat:
		features = np.concatenate([freq_feat, phase_feat, avg_feat, std_feat], axis=0)
		assert features.shape[-1] == data_norm.shape[-1], features.shape
		return features
	return [freq_feat, phase_feat, avg_feat, std_feat]  # shape for each: (feature_dim, num_sample)


def gen_features_reduce_dim(data: pd.DataFrame, dims,  power = 1, avg_scale = 1/1e5, std_scale = 1/1e5, *, concat = True):
	feat_list = gen_features(data, power, avg_scale, std_scale, concat=False)
	assert len(dims) == 4 == len(feat_list)

	for i, dim in enumerate(dims):
		pca = PCA(n_components=dim, svd_solver='full')
		feat_list[i] = pca.fit_transform(feat_list[i].T)

	if concat:
		features = np.concatenate(feat_list, axis=1)
		return features
	return feat_list


def gen_features_reduce_dim2(data: pd.DataFrame, dim, power = 1, avg_scale = 1/1e5, std_scale = 1/1e5):
	freq_feat, _, avg_feat, _ = gen_features(data, power, avg_scale, std_scale, concat=False)
	features = np.concatenate([freq_feat, avg_feat], axis=0).T

	pca = PCA(n_components=dim, svd_solver='full')
	features = pca.fit_transform(features)
	return features


def gen_features_reduce_dim3(data: pd.DataFrame, dim, power = 1, avg_scale = 1/1e5, std_scale = 1/1e5):
	freq_feat, _, avg_feat, _ = gen_features(data, power, avg_scale, std_scale, concat=False)
	features = np.concatenate([freq_feat, avg_feat], axis=0).T

	tsne = TSNE(n_components=dim)
	features = tsne.fit_transform(features)
	return features


def cluster_kmeans(features: np.array, k_list, num_experiment = 35):
	sil_df, inertia_df, labels_df = [], [], []

	for e in range(num_experiment):
		print(f"Experiment {e}")

		sil_list, inertia_list, label_list = [], [], []
		for k in k_list:
			# print(f"processing {k} cluster")
			cluster_model = KMeans(n_clusters=k, max_iter=1000)
			cluster_model.fit(features)
			sil = silhouette_score(features, cluster_model.labels_)

			sil_list.append(sil)
			inertia_list.append(cluster_model.inertia_)
			label_list.append(cluster_model.labels_)

		sil_df.append(sil_list)
		inertia_df.append(inertia_list)
		labels_df.append(label_list)

	sil_df = pd.DataFrame(sil_df, columns=k_list)
	inertia_df = pd.DataFrame(inertia_df, columns=k_list)
	labels_df = pd.DataFrame(labels_df, columns=k_list)

	return sil_df, inertia_df, labels_df


def agg_cluster_labels(labels: np.array, label_names):
	return (
		pd.Series(labels, index=label_names)
			.rename('agg clusters')
			.sort_values()
			.reset_index()
			.groupby('agg clusters')['index']
			.apply(list)
	)


