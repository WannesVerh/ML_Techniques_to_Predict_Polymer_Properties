import deepchem as dc
import numpy as np

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,RationalQuadratic, Matern

#load the dataset (polymer smiles and a specific task)
@ignore_warnings(category=ConvergenceWarning)
def main():
    tasks = ["PC", "RKTZRA", "SG", "TB", "TC", "ZC"]

    print("\nloading the data...")
    loader = dc.data.CSVLoader(tasks, feature_field="smiles", featurizer=dc.feat.RDKitDescriptors())
    Data = loader.create_dataset('./Alkane_dataset/csv_files/CombinedDataset.csv')
    #some RDKit descriptors return nan, make these 0
    X = np.nan_to_num(Data.X, copy=True, nan=0.0, posinf=0)

    for task in range(len(tasks)):
        #add data to dataset with the according propertie to be predicted
        Dataset = dc.data.DiskDataset.from_numpy(X=X, y=Data.y[:,task], w=Data.w[:,task], ids=Data.ids, tasks = [task])
        print("\n\npredicting", tasks[task],"...")

        #initiate lists to keep the results
        train_r2scores = []
        valid_r2scores = []
        test_r2scores = []
        RMSE_scores = []

        #initiate a loop of 20 times in order to get a decent estimation on the models performance
        for i in range(20):
            
            #split the dataset using the random splitter
            splitter = dc.splits.RandomSplitter()
            train_dataset, valid_dataset,  test_dataset = splitter.train_valid_test_split(Dataset, frac_train = 0.6, frac_valid = 0.2, frac_test = 0.2)
        

            # create the GPR model & fit the model
            kernel = 1 * RationalQuadratic()
            model = dc.models.SklearnModel(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5))

            #fit the model
            model.fit(train_dataset)
            

            #predict the test set
            predicted_test = model.predict(test_dataset)

            #calculate r2 scores
            metric = dc.metrics.Metric(dc.metrics.pearson_r2_score)
            train_r2score = model.evaluate(train_dataset, metric)
            valid_r2score = model.evaluate(valid_dataset, metric)
            test_r2score= model.evaluate(test_dataset, metric)

            #make then useable
            testr2=list(test_r2score.values())[0]
            validr2=list(valid_r2score.values())[0]
            trainr2=list(train_r2score.values())[0]

            #calculate RMSE score
            from sklearn.metrics import root_mean_squared_error
            RMSE_score = root_mean_squared_error(test_dataset.y,predicted_test)
            
            #add them to the list:
            train_r2scores.append(trainr2)
            valid_r2scores.append(validr2)
            test_r2scores.append(testr2)
            RMSE_scores.append(RMSE_score)

        #average the results and print to screen
        print("average training r2-score:",round(np.mean(train_r2scores),4 ))
        print("average valid r2-score:",round(np.mean(valid_r2scores),4) )
        print("average test r2-score:",round(np.mean(test_r2scores),4) )
        print("average test RMSE-score:",round(np.mean(RMSE_scores),4) )
    
main()