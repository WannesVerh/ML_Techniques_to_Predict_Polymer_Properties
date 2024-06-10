import deepchem as dc
import pandas as pd
import numpy as np
import csv
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import  RationalQuadratic, WhiteKernel


#the model that predicts the states
def PredictState(Smile, Ts, descriptors):

    #featurize the Smile string
    featurizer = dc.feat.RDKitDescriptors(descriptors=descriptors)
    RDKitFeats = featurizer.featurize(Smile)
    features = RDKitFeats.append(Ts)

    #load the model
    kernel = 1 * RationalQuadratic()
    Class_model = dc.models.SklearnModel(GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=1),model_dir = './T_dependent_density_database/Predcicting_unknown_samples/SavedModels')
    Class_model.reload()

    #Predict the state
    prediction = Class_model.predict(features)

    #the models predicts the probability on which state a certain molecule belongs to
    #the state with which the highest probability is correlated will be considered to be "the" state of the molecule
    state = np.argmax(prediction)
    print("The model predicted sate {state} for compound {Smile}.")

    return state

#the model that predicts the states
def PredictDensity_State0(Smile, Ts, descriptors):

    #featurize the Smile string
    featurizer = dc.feat.RDKitDescriptors(descriptors=descriptors)
    RDKitFeats = featurizer.featurize(Smile)
    features = RDKitFeats.append(Ts)

    #load the model
    kernel = 1 * RationalQuadratic() +WhiteKernel()
    State0_model = dc.models.SklearnModel(GaussianProcessClassifier(kernel=kernel, n_restarts_optimizer=1),model_dir = './T_dependent_density_database/Predcicting_unknown_samples/SavedModels')
    State0_model.reload()

    #Predict the state
    prediction = State0_model.predict(features)

    #the models predicts the probability on which state a certain molecule belongs to
    #the state with which the highest probability is correlated will be considered to be "the" state of the molecule
    density  = round(prediction,2)
    print("The model predicted a density of {density} for compound {Smile}.")

    return density


def main():

    #first load the rigth set of RDKit descriptors for each state/model
    print("Loading RDKit descriptors...")
    
    desc_class = pd.read_csv('./T_dependent_density_database/csv_files/classification_feats.csv')
    Classdescriptors = desc_class["descriptors"].to_list()

    desc_s0 = pd.read_csv('./T_dependent_density_database/csv_files/State0_feats.csv')
    State0descriptors = desc_s0["descriptors"].to_list()

    desc_s1 = pd.read_csv('./T_dependent_density_database/csv_files/State1_feats.csv')
    State1descriptors = desc_s1["descriptors"].to_list()

    desc_s2 = pd.read_csv('./T_dependent_density_database/csv_files/State2_feats.csv')
    State2descriptors = desc_s2["descriptors"].to_list()

    #then load the samples that must be predicted
    with open('./T_dependent_density_database/Predcicting_unknown_samples/UnknownSamples.csv','r') as file:
        reader = csv.reader(file, delimiter = ',')
        Smiles = []
        Ts = []

        for idx,row in enumerate(reader):
            if idx != 0:
                Smiles.append(row[0])
                Ts.append(row[1])

    # Now loop over the smile strings and make predictions for every        
    for i, molecule in enumerate(Smiles):

        #start by predicting the state
        #State = PredictState(molecule, Ts[i], Classdescriptors)
        State = 0
        if State == 0:
            dens = PredictDensity_State0(molecule, Ts, State0descriptors)
        elif State == 1:
            dens = 1
        elif State == 2:
            dens = 2

main()