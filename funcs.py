from scipy.cluster import hierarchy as hc
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def model_fit(model, X_train, y_train, X_val, y_val):
	model.fit(X_train, y_train)
	preds_train = model.predict(X_train)
	preds_val = model.predict(X_val)
	rmse_train = mean_squared_error(y_train, preds_train, squared=False)
	rmse_val = mean_squared_error(y_val, preds_val, squared=False)
	r2_train = r2_score(y_train, preds_train)
	r2_val = r2_score(y_val, preds_val)

	result = pd.DataFrame({"rmse_train":rmse_train, "rmse_val":rmse_val, "r2_train":r2_train, "r2_val":r2_val}, index=["metrics"])
	print(result)


def get_perm_imp(model, x, y):
	imp = permutation_importance(model, x, y, scoring="neg_root_mean_squared_error", n_repeats=2, n_jobs=-1, random_state=42)
	df_pi = pd.DataFrame({"features":x.columns, "imp":imp.importances_mean}, index=None).sort_values(by="imp", ascending=False).reset_index(drop=True)
	return df_pi


def plot_pi(pi_res, limit=None):
	if limit is None:
		limit = len(pi_res)
	pi_res_lim = pi_res.iloc[:limit]
	pi_res_lim.plot.barh(x="features", y="imp", figsize=(10,6))
	plt.title("Permutation Importances")
	plt.show()
	

def get_fi(model, x):
	return pd.DataFrame(np.stack([x.columns, model.feature_importances_], axis=1), columns=["features", "imp"]).sort_values(by="imp", ascending=False).reset_index(drop=True)


def plot_fi(fi_res, limit=None):
	if limit is None:
		limit = len(fi_res)
	fi_res_lim = fi_res.iloc[:limit]
	fi_res_lim.plot.barh(x="features", y="imp", figsize=(10,6))
	plt.title("Feature Importances")
	plt.show()


def optimum_feats(model, X_train, X_val, y_train, y_val, n_feats):
	errors = []
	pi = get_perm_imp(model, X_val, y_val)
	for n in n_feats:
		rf = RandomForestRegressor()
		f = pi[:n].features.tolist()
		rf.fit(X_train.loc[:,f], y_train)
		preds_val = rf.predict(X_val.loc[:,f])
		errors.append(mean_squared_error(y_val, preds_val))

	plt.plot(n_feats, errors)
	plt.show()

		



if __name__ == "__main__":
	model_fit()
	get_perm_imp()
	plot_pi()
	get_fi()
	plot_fi()
	optimum_feats()