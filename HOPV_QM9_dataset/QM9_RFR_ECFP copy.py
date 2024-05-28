#import the most used things
print("importing...")
import deepchem as dc
import numpy as np
import pandas as pd

#load the qm9 dataset and use rdkit descriptors, randomly split the data in training, validation en test
print("loading...")
tasks, data, transformers = dc.molnet.load_qm9(featurizer=dc.feat.CircularFingerprint(), splitter = None)


#some RDKit descriptors return nan, make these 0 and select the prefered property
#QM9 has the following properties: ["mu", "alpha", "homo", "lumo", "gap", "r2", "zpve", "cv", "u0", "u298", "h298", "g298"]
# we want the "homo" property so idx = 2

#convert the dataset so that no more nan is present and selct the y-values of the homo gap toghether with the according wheigths
X = np.nan_to_num(data[0].X, copy=True, nan=0.0)
Y = data[0].y[:,[2]]
W = data[0].w[:,[2]]

#add the converted data to new dataframe
new_data= dc.data.DiskDataset.from_numpy(X=X, y=Y, w=W, ids=data[0].ids, tasks = ["HOMO"])

#split the data in train and test samples
splitter = dc.splits.RandomSplitter()
train_dataset, test_dataset = splitter.train_test_split(new_data)



# create the randomforest model & fit the model
from sklearn.ensemble import RandomForestRegressor

model = dc.models.SklearnModel(RandomForestRegressor(n_estimators=500))

print("fitting model...")
model.fit(train_dataset)
print("model is fitted")

#predict the test set
predicted = model.predict(test_dataset)



#calculate r2 score
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
print('Training set score:', model.evaluate(test_dataset, metric))
test_score= model.evaluate(test_dataset, metric)
print('Test set score:',test_score )


#convert tekst to string, so it can be depicted in matplotlib
number=list(test_score.values())[0] #test score is returned by a dict so convert into array an take the 0th element
text = "r2= "+ str(round(number,4)) #round til 4 places after the comma and make a string

#plot the data
import matplotlib.pyplot as plt
x=[-5,10]
y=[-5,10]
plt.plot(x, y, linestyle="dashed")
plt.scatter(test_dataset.y, predicted, label="Observations")
plt.legend()
plt.xlabel("experimental homo gap (Hartree)")
plt.ylabel("ML predicted homo gap (Hartree)")
plt.title("QM9_dataset_RFR_CircularFingerprint")
plt.text(-4.5,8,text)
plt.savefig("./images/QM9_dataset_RFR_CircularFingerprint.png")