import deepchem as dc
import numpy as np
import timeit
import pandas as pd

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,RationalQuadratic, Matern, WhiteKernel
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from numpy import average



#function to get the names of the descriptors based on the support vector
def Get_Selected_features(support,descriptorlist):
    for idx, bool in enumerate(support):
        if bool == True:
            print(descriptorlist[idx])



@ignore_warnings(category=ConvergenceWarning)
def main():

    #generate the list of RDKit descriptors
    from rdkit.Chem import Descriptors
    descriptors = Descriptors._descList

    #make a list with the names of descriptors available on RDKit
    descriptorlist = []
    for item in descriptors:
        descriptorlist.append(item[0])

    #this is done since RDKit utility in deepchem calculates the descriptors in alphabetical order,
    # this way we can later extract the used descriptors
    descriptorlist = sorted(descriptorlist)

    ##################################################
    ############ change ##############################
    task = "density"
    ##################################################
    ##################################################


    file = './T_dependent_density_database/csv_files/RoomTemp_dens_Dataset.csv'

    print("\nloading the data...")
    loader = dc.data.CSVLoader([task], feature_field="smiles", featurizer=dc.feat.RDKitDescriptors())
    Data = loader.create_dataset(file)


    #some RDKit descriptors return nan, make these 0
    X = np.nan_to_num(Data.X, copy=True, nan=0.0, posinf=0)
    Dataset = dc.data.DiskDataset.from_numpy(X=X, y=Data.y, w=Data.w, ids=Data.ids, tasks = [task])
    print("\n\npredicting",task,"...")

    
    numbers = [5,10,15,20,30,40,50,60,70,100,150,210]
    
    for n_of_feats in numbers:
        print("\n\nnumber of feats:",n_of_feats)

        #select the features using a randomforest (decisionTreeRegressor)
        selector = RFE(estimator=DecisionTreeRegressor(), n_features_to_select=n_of_feats, step = 150)
        X_Selected = selector.fit_transform(Dataset.X, Dataset.y.ravel()) #y is a colum vector but it needs a 1d array -> ravel fixes this
        RFEDataset = dc.data.DiskDataset.from_numpy(X=X_Selected, y=Dataset.y, w=Dataset.w, ids=Dataset.ids, tasks = [task])

        #find which descriptors are the most important
        selected = selector.support_ #this returns an array with true and false, True are the selected features
        Get_Selected_features(selected,descriptorlist)
        
        
        #initiate lists to keep the results
        train_r2scores = []
        valid_r2scores = []
        test_r2scores = []
        RMSE_scores = []
        start = timeit.default_timer()


        #initiate a loop of 10 times in order to get a decent estimation on the models performance
        for i in range(10):
            
            #split the dataset using the random splitter
            splitter = dc.splits.RandomSplitter()
            train_dataset, valid_dataset,  test_dataset = splitter.train_valid_test_split(RFEDataset, frac_train = 0.6, frac_valid = 0.2, frac_test = 0.2)
        

            # create the GPR model & fit the model
            kernel = 1 * RationalQuadratic() + WhiteKernel()
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
            
            #add them to the list:
            train_r2scores.append(trainr2)
            valid_r2scores.append(validr2)
            test_r2scores.append(testr2)
            RMSE_scores.append(RMSE_score)

        stop = timeit.default_timer()

        #average the results and print to screen
        print("average training r2-score:",round(np.mean(train_r2scores),4))
        print("average valid r2-score:",round(np.mean(valid_r2scores),4) )
        print("average test r2-score:",round(np.mean(test_r2scores),4 ))
        print("average test RMSE-score:",round(np.mean(RMSE_scores),4) )
        print("Time:",stop-start)
    
main()