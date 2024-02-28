import csv
import pandas as pd
from sympy import false

NewData = {}
dataChuncks = [390, 432, 3380, 561, 368, 370, 382, 382] # this indicates how much polymers are used for the property
start = 1
property = 0



for chunck in dataChuncks:
    
    with open("./csv_files/export.csv","r") as file: #open the original dataset to read
        df=pd.read_csv(file, sep=',',skiprows=range(1,start),nrows=chunck) #read chuck by chunk in order to avaoid a full memory

        data = df.values.tolist() #change pandas dataframe to a list

        #loop over every datarow that is in the specific chunck
        for row in (data):#if this polymer is already present just change the zero to the actual value
            if row[2]== 'Eat':
                NewData[row[1]]= float(row[3]) # add the polymer as key and add the Ei value

    property = property +1 # enter the value in the next place of the array
    start = start + chunck  #set the startpoint equal to endpoint of previous chuck


#write the data to a new file
with open('./csv_files/Polymers_Eat.csv', 'w', newline='') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(['smiles','Eat'])
    for key, val in NewData.items():
        Toadd = [key,val]
        writer.writerow(Toadd)
   

