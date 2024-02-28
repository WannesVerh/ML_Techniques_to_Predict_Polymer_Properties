import deepchem as dc
import numpy as np


#making the data
smiles = [
    'C(COCCOCCOCCOCCO)O',
    'C(COCCOCCOCCOCCOCCOCCOCCO)O',
    'C(COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO)O',
    'C(COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO)O',
    'C(COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO)O',
    'C(COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO)O',
    'C(COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO)O',

]
properties = np.array([77.4, 105, 133, 160, 188, 216, 234]) #TPSA values
featurizer = dc.feat.CircularFingerprint(size=5)
print("for the 7 input smiles, the featurizer CircularFingerprint is used with size 1024:")
ecfp = featurizer.featurize(smiles)
print("shape of featurized data: ",ecfp.shape)

dataset = dc.data.NumpyDataset(ecfp, properties)
print("dataset will look like:")
print(dataset.to_dataframe())

#split the data 
splitter = dc.splits.RandomSplitter()
# split 5 datapoints in the ratio of train:valid:test = 3:1:1
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(
  dataset=dataset, frac_train=0.6, frac_valid=0.2, frac_test=0.2
)


#model
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
model = dc.models.SklearnModel(model=rf)
# model training
model.fit(train_dataset)
valid_preds = model.predict(valid_dataset)
print(valid_preds, valid_dataset.y, valid_dataset.ids)


"""###test
SampleToPredict = 'C(COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO)O'
featurizedSample = featurizer.featurize(SampleToPredict)
print(featurizedSample.shape)
TPSA_Sample = model.predict(featurizedSample)
print(TPSA_Sample)"""


"""# initialze the metric
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
# evaluate the model
train_score = model.evaluate(train_dataset, [metric])
print("training score: ",train_score)
valid_score = model.evaluate(valid_dataset, [metric])
print("valid score: ",valid_score)
test_score = model.evaluate(test_dataset, [metric])
print("test score: ",test_score)
print(test_dataset.X)"""