import pandas as pd
from collections import defaultdict


def _label_outliers_along_series(s_win: pd.Series, r) -> list:
	outliers_idx = []
	while True:
		abs_norm_s_win = abs((s_win - s_win.mean()) / s_win.std())
		outlier_s: pd.Series = abs_norm_s_win[abs_norm_s_win > r]
		if outlier_s.empty:
			break
		outlier_idx = outlier_s.idxmax()
		outliers_idx.append(outlier_idx)
		s_win = s_win.drop(outlier_idx)
	return outliers_idx


def label_outliers(df: pd.DataFrame, r, win_len) -> dict:
	outliers_coords = defaultdict(list)
	for i in range(0, df.shape[0], win_len):
		win = df.iloc[i: i + win_len, :]
		for col, s_win in win.items():
			outliers_idx = _label_outliers_along_series(s_win, r)
			outliers_coords[col] += outliers_idx
	return outliers_coords
