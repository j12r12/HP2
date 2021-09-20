from scipy.cluster import hierarchy as hc
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt

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
	plt.show()
	









if __name__ == "__main__":
	model_fit()
	get_perm_imp()
	plot_pi()