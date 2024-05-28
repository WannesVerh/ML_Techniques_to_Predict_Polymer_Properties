import deepchem as dc
import numpy as np
import csv

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,RationalQuadratic

def DatasetGenerator():
    #first we read the csv file to featurize oligomer 1:
    #descriptors to use are:
    descriptors = ["BCUT2D_CHGLO",
    "BCUT2D_MRLOW",
    "Chi0",
    "MaxAbsPartialCharge",
    "NumHAcceptors"
    ]

    with open('./Polyol_mixtures/csv_files/OURDATASET.csv','r') as file:
        reader = csv.reader(file, delimiter = ',')
        X = []
        y = []
        Ids = []
        #define out featurizer with the 5 descriptors
        featurizer = dc.feat.RDKitDescriptors(descriptors=descriptors)
        for idx, row in enumerate(reader):
            if row[0] != "Oligomer1":  #skip the first header row
                frac =  float(row[2])
                
                #calculate descriptors for the oligomers
                feat1 = featurizer.featurize(row[0])
                feat2 = featurizer.featurize(row[1])

                """since these feats are only present for a fraction we multiply the feats of oligomer 1 with the according fraction it is present,
                the same will be done for oligomer 2"""
                
                feat1 = feat1 * frac
                feat2 = feat2 * (1-frac)

                #the feats are returned as np culumns so we ravel them
                feat1 = feat1.ravel()
                feat2 = feat2.ravel()
                X_row = np.append(feat1,feat2)

                #add the combined array
                X.append(X_row)

                #add the density
                y.append(float(row[3]))

                #now generate an ID of the mixture:
                id = row[0] + "|" + row[1] + "|" + row[2]
                Ids.append(id)
    X_data = np.nan_to_num(X, copy=True, nan=0.0)
    #both y and ids are rows but we need columns


    #add data to dataset
    Dataset = dc.data.DiskDataset.from_numpy(X=X_data, y=y, ids=Ids, tasks = ["Density"])

    return Dataset


#read the csv file
print('loading the data...')
Dataset = DatasetGenerator()


#split the dataset using the random splitter
splitter = dc.splits.RandomSplitter()
train_dataset, test_dataset = splitter.train_test_split(Dataset)
print("Data is splitted into: train, valid, test")

# create the GPR model & fit the model
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic

kernel = 1 * RBF()
model = dc.models.SklearnModel(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10))

print("fitting model...")
model.fit(train_dataset)
print("model is fitted")

#predict the test set
predicted = model.predict(test_dataset)

#calculate r2 score
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
print('Training set score:', model.evaluate(train_dataset, metric))
test_score= model.evaluate(test_dataset, metric)
print('Test set score:',test_score )

#calculate MSE score
metric = dc.metrics.Metric(dc.metrics.mean_squared_error)
print('Training set score:', model.evaluate(train_dataset, metric))
test_RMSE_score = model.evaluate(test_dataset, metric)
print('Test set score:', test_RMSE_score)

#plot the data
import matplotlib.pyplot as plt


#convert tekst to string, so it can be depicted in matplotlib
r2_number=list(test_score.values())[0]
rmse_number = list(test_RMSE_score.values())[0]
text = "r2= "+ str(round(r2_number,5)) + " & RMSE= "+ str(round(rmse_number,5))

x=[1,1.3]
y=[1,1.3]
#text = "R2= "+ str(round(test_score))
plt.plot(x, y, linestyle="dashed")
plt.scatter(test_dataset.y, predicted, label=text)
plt.legend()
plt.xlabel("experimental density (g/cm^3)")
plt.ylabel("ML predicted density (g/cm^3)")
plt.title("GPR(RBF) with 5 RDkit descriptors density of mixed oligomers")
plt.savefig('./Polyol_mixtures/figures/Our_dataset_mixture_densities_GPR_RBF.png')