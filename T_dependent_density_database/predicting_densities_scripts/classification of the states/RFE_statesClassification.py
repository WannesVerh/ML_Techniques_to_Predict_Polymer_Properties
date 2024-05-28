import deepchem as dc
import numpy as np
import timeit
import pandas as pd

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF,RationalQuadratic, Matern, WhiteKernel
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
from numpy import average

#function to get the results from the classification
def GetScore(dataset, name_of_set,model):
    #predict the specific dataset
    pred_dataset = model.predict(dataset)

    #get the real values
    y_true = dataset.y.ravel()

    y_pred = []
    for estimations in pred_dataset:
        #this classifier returns a change of which state the molecule will be, so we get the highest one 
        index_maxchance = np.argmax(estimations)
        y_pred.append(index_maxchance)

    #calculate the accuracy:
    from sklearn.metrics import recall_score

    score = recall_score(y_true, y_pred, average='micro')
    #print(f"the model predicted the {name_of_set} with an accuracy of {round(score*100,2)}% ")
    return score, y_pred

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
    task = "state"
    ##################################################
    ##################################################


    file = './T_dependent_density_database/csv_files/RoomTemp_dens_Dataset_WithState.csv'

    print("\nloading the data...")
    loader = dc.data.CSVLoader([task], feature_field="smiles", featurizer=dc.feat.RDKitDescriptors())
    Data = loader.create_dataset(file)


    #some RDKit descriptors return nan, make these 0
    X = np.nan_to_num(Data.X, copy=True, nan=0.0, posinf=0)
    Dataset = dc.data.DiskDataset.from_numpy(X=X, y=Data.y, w=Data.w, ids=Data.ids, tasks = [task])
    print("\n\npredicting",task,"...")

    
    numbers = [70,100,210]
    
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
        start = timeit.default_timer()


        #initiate a loop of 10 times in order to get a decent estimation on the models performance
        for i in range(10):
            
            #split the dataset using the random splitter
            splitter = dc.splits.RandomSplitter()
            train_dataset, valid_dataset,  test_dataset = splitter.train_valid_test_split(RFEDataset, frac_train = 0.8, frac_valid = 0.1, frac_test = 0.1)
        

            # create the GPR model & fit the model
            kernel = 1 * RBF(1.0)
            model = dc.models.SklearnModel(GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=5))

            #fit the model
            model.fit(train_dataset)
            

            #evaluate the model
            train_score, train_pred = GetScore(train_dataset,"train_dataset",model)
            valid_score, valid_pred = GetScore(valid_dataset, "valid_dataset",model)
            test_score, test_pred = GetScore(test_dataset, "test_dataset",model)

            
            #add them to the list:
            train_r2scores.append(train_score)
            valid_r2scores.append(valid_score)
            test_r2scores.append(test_score)


        stop = timeit.default_timer()

        #average the results and print to screen
        print("average training precision-score:",round(np.mean(train_r2scores*100),2),"%")
        print("average valid precision-score:",round(np.mean(valid_r2scores*100),2),"%" )
        print("average test precision-score:",round(np.mean(test_r2scores*100),2 ),"%")
        print("Time:",stop-start)
    
main()