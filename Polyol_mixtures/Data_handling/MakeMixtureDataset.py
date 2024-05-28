import pandas as pd
import numpy as np
import csv

from sympy import frac 

Pgs_csv = './Polyol_mixtures/csv_files/dens_PG.csv'
Egs_csv = './Polyol_mixtures/csv_files/dens_EG.csv'

#first fetch the necc data for the ethyleneglycols
Egs_smiles = []
Egs_dens = []
with open(Egs_csv,'r') as file:
    reader = csv.reader(file, delimiter = ',')
    
    for idx, row in enumerate(reader):
        if row[1] != "smiles":  #skip the first header row
            Egs_smiles.append(row[1])
            Egs_dens.append(float(row[3]))


#now fetch the data from the propyleneglycols
Pgs_smiles = []
Pgs_dens = []
with open(Pgs_csv,'r') as file:
    reader = csv.reader(file, delimiter = ',')
    
    for idx, row in enumerate(reader):
        if row[1] != "smiles":  #skip the first header row
            Pgs_smiles.append(row[1])
            Pgs_dens.append(float(row[3]))
    

#now make the mixtures add the fraction and the corresponding density

fractions = [0.1, 0.3, 0.5, 0.7, 0.9]
Data = []

"""Loop over the first oligomer list, then over the second to make all oligomer combinations.
Per combination also make a mixture with a certain fraction of oligomers"""
for idx1, oligomer1 in enumerate(Egs_smiles):
    for idx2, oligomer2 in enumerate(Pgs_smiles):
        for fraction in fractions:
            density = fraction*Egs_dens[idx1] + (1-fraction)*Pgs_dens[idx2]
            to_add = [oligomer1, oligomer2, fraction, density]
            Data.append(to_add)

with open('./Polyol_mixtures/csv_files/OURDATASET2.csv', 'w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Oligomer1', 'Oligomer2', 'Frac1', 'Density'])   

    for row in Data:    
        writer.writerow(row)

