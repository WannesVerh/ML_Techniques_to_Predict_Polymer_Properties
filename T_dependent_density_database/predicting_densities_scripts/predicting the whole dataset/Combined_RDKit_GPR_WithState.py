import deepchem as dc
import pandas as pd
import numpy as np

#load the dataset (polymer smiles and their specific gravity)

descriptors = ["BCUT2D_LOGPLOW",
"BCUT2D_MRHI",
"Ipc",
"SlogP_VSA4",
"SlogP_VSA6",
]

print("loading the data...")
loader = dc.data.CSVLoader(["density"], feature_field="smiles", featurizer=dc.feat.RDKitDescriptors(descriptors = descriptors))
Data = loader.create_dataset('./T_dependent_density_database/csv_files/Combined_dens_Dataset_WithState.csv')
print("data loaded!")


#some RDKit descriptors return nan, make these 0
X = np.nan_to_num(Data.X, copy=True, nan=0.0, posinf=0)
print("RDKit:",X.shape)

# now load the additional features
df = pd.read_csv('./T_dependent_density_database/csv_files/RoomTemp_dens_Dataset_WithState.csv')
state = df["state"].to_numpy()
Ts = df["temp"].to_numpy()

# combine the RDKit descriptors with the simulation temperature and the boiling temperature
input=np.column_stack((X,Ts,state))
print("With TS and TB added:",input.shape)

#add data to dataset
Dataset = dc.data.DiskDataset.from_numpy(X=input, y=Data.y, w=Data.w, ids=Data.ids, tasks = ["denstiy"])

#split the dataset using the random splitter
splitter = dc.splits.RandomSplitter()
train_dataset, valid_dataset, test_dataset = splitter.train_valid_test_split(Dataset)
print("Data is splitted into: train, valid, test")

# create the GPR model & fit the model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, DotProduct, ConstantKernel, Matern, RationalQuadratic, ExpSineSquared, DotProduct, WhiteKernel

kernel = 1 * RationalQuadratic() +WhiteKernel()
model = dc.models.SklearnModel(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10))

print("fitting model...")
model.fit(train_dataset)
print("model is fitted")

#predict the test set
predicted = model.predict(test_dataset)

#calculate r2 scores
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
train_r2score = model.evaluate(train_dataset, metric)
valid_r2score = model.evaluate(valid_dataset, metric)
test_r2score= model.evaluate(test_dataset, metric)

#make then useable
testr2=list(test_r2score.values())[0]
validr2=list(valid_r2score.values())[0]
trainr2=list(train_r2score.values())[0]

print("training r2-score:",np.mean(trainr2) )
print("valid r2-score:",np.mean(validr2) )
print("test r2-score:",np.mean(testr2) )

#calculate RMSE score
from sklearn.metrics import root_mean_squared_error
RMSE_score = root_mean_squared_error(test_dataset.y,predicted)
print('tets set score:',RMSE_score)

#convert tekst to string, so it can be depicted in matplotlib
text = "test r2= "+ str(round(testr2,4))  + " & RMSE= " +str(round(RMSE_score,4))

#plot the data
import matplotlib.pyplot as plt

x=[0,1000]
y=[0,1000]
#text = "R2= "+ str(round(test_score))
plt.plot(x, y, linestyle="dashed")
plt.scatter(test_dataset.y, predicted, label=text)
plt.legend()
plt.xlabel("density (kg/m^3)")
plt.ylabel("ML predicted density (kg/m^3)")
plt.title("GPR(RQ+White) to predict the density of hydrocarbons, with TB added")
plt.show()
