import numpy as np
import deepchem as dc
from sklearn.gaussian_process import GaussianProcessRegressor

n_features = 1024
hopv_tasks, hopv_datasets, transformers = dc.molnet.load_hopv()
train_data, valid_data, test_data = hopv_datasets

metric = [dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean, mode="regression"),
          dc.metrics.Metric(dc.metrics.mean_absolute_error, np.mean, mode="regression")]

model = dc.models.SklearnModel(GaussianProcessRegressor())

model.fit(train_data)

print("Evaluating model")
train_scores = model.evaluate(train_data, metric, transformers)
valid_scores = model.evaluate(valid_data, metric, transformers)

print("Train scores")
print(train_scores)

print("Validation scores")
print(valid_scores)


