import deepchem as dc
import numpy as np
import pandas as pd

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,RationalQuadratic, Matern, WhiteKernel

#load the dataset (polymer smiles and a specific task)
@ignore_warnings(category=ConvergenceWarning)
def main():
    #tasks = ["PC", "RKTZRA", "SG", "TB", "TC", "ZC"]
    tasks = ["ZC"]
    #generate a list to fix graph titles:
    names = {"PC":"Critical pressure [Bar]",
              "RKTZRA":"liquid molar volume [m3/Kmol]", 
              "SG":"Specific gravitation [au]", 
              "TB":"Boiling temperature [°C]",
              "TC":"Critical temperature [°C]",
              "ZC":"critical compressibility factor [au]" }

    # generate the coordinated of the diagonal line:
    coordinate = {"PC" : [0,60], "RKTZRA":[0.15,0.35], "SG":[0.3,1], "TB":[-250,800], "TC":[-100,800], "ZC":[0.1,0.4]}
    # read the csv file containening the neccesarry rdkit descriptors
    feats = pd.read_csv('./Polyethylenes/csv_files/RFE_features.csv')
    

    for task in tasks:
        # read the dataset belonging to the rigth property
        print(task)
        file = './Polyethylenes/csv_files/Combined_' + task + '.csv'

        #get the descriptors neccesarry for the property
        descriptors = feats[task]
        descriptorlist = descriptors.dropna().to_list()

        print("\nloading the", task, "data...")
        loader = dc.data.CSVLoader([task], feature_field="smiles", featurizer=dc.feat.RDKitDescriptors(descriptorlist))
        Data = loader.create_dataset(file)
        #some RDKit descriptors return nan, make these 0
        X = np.nan_to_num(Data.X, copy=True, nan=0.0, posinf=0)

        #add data to dataset with the according propertie to be predicted
        Dataset = dc.data.DiskDataset.from_numpy(X=X, y=Data.y, w=Data.w, ids=Data.ids, tasks = [task])
        print("\n\npredicting", task,"...")

        #initiate lists to keep the results
        train_r2scores = []
        valid_r2scores = []
        test_r2scores = []
        RMSE_scores = []

        #initiate a loop of 10 times in order to get a decent estimation on the models performance
        for i in range(2):
            
            #split the dataset using the random splitter
            splitter = dc.splits.RandomSplitter()
            train_dataset, valid_dataset,  test_dataset = splitter.train_valid_test_split(Dataset, frac_train = 0.8, frac_valid = 0.1, frac_test = 0.1)
        

            # create the GPR model & fit the model
            kernel = 1 * RationalQuadratic() + 1* WhiteKernel()
            model = dc.models.SklearnModel(GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10))

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

            print("round", i ,":  train-r2:",round(trainr2,3),"valid-r2:", round(validr2,3), "test-r2:", round(testr2,3), "test-rmse:", round(RMSE_score,4))
            #add them to the list:
            train_r2scores.append(trainr2)
            valid_r2scores.append(validr2)
            test_r2scores.append(testr2)
            RMSE_scores.append(RMSE_score)
        
            if i == 0:
                #for the first run generate a graph
                import matplotlib.pyplot as plt

                x=coordinate[task]
                y=coordinate[task]

                text = "R2= "+ str(round(testr2,4))
                title = "GPR(RQ + White) & RFE & [80%,10%,10%] for " + task
                x_text = "known "+ names[task]
                y_text = "ML predicted "+ names[task]
                save = './Alkane_dataset/images/GPR_RQ_White_RFE_' + task + ".png"
                plt.plot(x, y, linestyle="dashed")
                plt.scatter(test_dataset.y, predicted_test, label=text)
                plt.legend()
                plt.xlabel(x_text)
                plt.ylabel(y_text)
                plt.title(title)
                plt.savefig(save)
                plt.close()
        #average the results and print to screen
        print("average training r2-score:",round(np.mean(train_r2scores),3))
        print("average valid r2-score:",round(np.mean(valid_r2scores),3) )
        print("average test r2-score:",round(np.mean(test_r2scores),3) )
        print("average test RMSE-score:",round(np.mean(RMSE_scores),4) )
    
    return None


main()