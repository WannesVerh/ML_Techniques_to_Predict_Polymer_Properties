import numpy as np
import pandas as pd
import csv 


#####
###change !!
density_eg = 1.11 #g/cm^3
mw_eg = 62.03678
path = './Polyol_mixtures/csv_files/ethyleneglycol_oligomers_100.csv'
#####

V_eg = mw_eg/density_eg
dVol = 17.58 #cm^3/mol diffrence in volume( equals volume of 2 hydrogen atoms and 1 oxygen)

densities = [density_eg]

with open(path,'r') as file:
    reader = csv.reader(file, delimiter = ',')
    for idx, row in enumerate(reader):

        #firstline
        if idx == 0: 
            continue

        #monomer
        elif idx == 1: 
            continue

        #oligomers
        else:
            Mw = float(row[2])
            units = int(row[0])
            volume = V_eg+ (units-1)*(V_eg-dVol)
            densities.append(Mw/volume)

df = pd.read_csv(path)
df['Density'] = densities

#change path!
df.to_csv('./OUR_DATASET/csv_files/dens_PG_100.csv', index=False)