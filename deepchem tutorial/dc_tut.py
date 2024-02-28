from cgi import test
import deepchem as dc
import numpy as np
import pandas as pd

N_samples = 50
N_features = 120
X=np.random.rand(N_samples,N_features)
Y=np.random.rand(N_samples)

dataset = dc.data.NumpyDataset(X,Y)
print(dataset.X.shape)
print(dataset.y.shape)

dataset2 = dc.data.DiskDataset.from_numpy(X,Y)
print(dataset2.X.shape)
print(dataset2.y.shape)

#now lets look at featurization
print("featurization")
smiles = [
    'C(CO)O',
    'C(COCCOCCOCCOCCO)O',
    'C(COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO)O',
    'C(COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO)O',
    'C(COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCO)O'
    ]

properties = [80, 74.25, 104.24, 130.2, 160.78]
featurizer = dc.feat.CircularFingerprint(size=1024)
feats = featurizer.featurize(smiles)
print(feats.shape)
dataset3 = dc.data.DiskDataset.from_numpy(np.array(feats), np.array(properties))
print(dataset3.X.shape)
print(dataset3.y.shape)


#now load data from a csv file

df = pd.DataFrame(list(zip(smiles, properties)), columns=["SMILES", "property"])
import tempfile
with dc.utils.UniversalNamedTemporaryFile(mode ='w') as tempfile:
    df.to_csv(tempfile.name)

    featurizer = dc.feat.CircularFingerprint(size = 1024)
    loader = dc.data.CSVLoader(["property"], feature_field="SMILES", featurizer=featurizer)
    dataset4 = loader.create_dataset(tempfile.name)
    print(len(dataset4))


#datasplitting

splitter =  dc.splits.RandomSplitter()
trainData, validData, testData = splitter.train_valid_test_split(dataset=dataset3, frac_train=0.6, frac_valid=0.2, frac_test=0.2)
print([len(trainData),len(validData), len(testData)])

    
from sklearn.gaussian_process import GaussianProcessRegressor
GPR = GaussianProcessRegressor()
model = dc.models.SklearnModel(model = GPR)

print("fitting model")
model.fit(trainData)

valid_pred = model.predict(validData)
print("real value: ",validData.y)
print("predicted value; ",valid_pred)

test_pred = model.predict(testData)
print("real value: ",testData.y)
print("predicted value; ",test_pred)

#validate the model
metric = dc.metrics.Metric(dc.metrics.mae_score)

trainScore = model.evaluate(trainData,[metric])
print("trainscore: ", trainScore)

testScore = model.evaluate(testData, [metric])
print("testscore: ", testScore)