import numpy as np
import pandas as pd


def find_elem_loc(df: pd.DataFrame, return_iloc = True) -> list:
	df = ~pd.isna(df)
	id_coords = list(zip(*np.where(df)))
	if return_iloc:
		return id_coords
	return [
		(df.index[i], df.columns[j])
		for i, j in id_coords
	]
