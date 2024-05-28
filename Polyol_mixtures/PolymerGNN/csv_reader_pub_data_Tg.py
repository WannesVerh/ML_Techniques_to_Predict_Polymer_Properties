import csv
import pandas as pd
import numpy as np

NewData = {}
dataChuncks = [390, 432, 3380, 561, 368, 370, 382, 382] # this indicates how much polymers are used for the property
start = 1
property = 0




    
with open("./csv_files/pub_data.csv","r") as file: #open the original dataset to read
    df=pd.read_csv(file, sep=',')
    data = df.values.tolist() #change pandas dataframe to a list
    Tg = []
    Iv = []
    features =[]
    #the fetaures present in the file a
    for datapoint in data:
        Tg.append(datapoint[9]) #the tg value can be found at idx 9
        Iv.append(datapoint[7])#the intrinsic viscosity can be found at index 7
        #selected features are: AN->3 | OHN->4 | OHN(Prim)->5 | OHN(sec)->6 | Mn(Ps)->12 | Mw(PS)->13 | Mz(PS)->14 | PDI(PS)->15 | Mn(abs)->16 | Mw(abs)->17 | Mz(abs)->18 | PDI(abs)->19
        get =[3, 4, 5, 6, 12, 13, 14, 15, 16, 17,18,19]
        features = np.array(datapoint)[get]
        print()
    """#loop over every datarow that is in the specific chunck
    for row in (data):#if this polymer is already present just change the zero to the actual value
        if row[1] in NewData:
            NewData[row[1]][property] = float(row[3])

        else: # add the polymer as key and make a list of zeros as the values
            NewData[row[1]]= [0]*len(dataChuncks)
            NewData[row[1]][property] = float(row[3])"""


'''#write the data to a new file
with open('TgData.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    for key, val in NewData.items():
        Toadd = [key]
        for data in val:
            Toadd.append(data)
        writer.writerow(Toadd)'''
   

