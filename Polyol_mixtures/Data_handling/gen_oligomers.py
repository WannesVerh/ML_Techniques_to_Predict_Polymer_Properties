from rdkit import Chem
import csv
from rdkit.Chem.Descriptors import ExactMolWt

def oligomerGenerator (monomer, RepeatingUnits):
    """Generate a function that generates oligomers for easy monomers based on their smiles string
    The idea is to just repeat the part that needs to be added,
    for simple molecules like ethyleneglycol (OCCO), this is everything except the fist atom
    """
    to_add = monomer[1:]
    oligomer = monomer+to_add*(RepeatingUnits-1) #add minus 1 becaus the strating unit is also present

    #check if generated string makes sense
    new_mol = Chem.MolFromSmiles(oligomer,sanitize = False)
    if new_mol is None:
        print('invalid SMILES')
        return None
    else:
        try:
            Chem.SanitizeMol(new_mol) #sanitization generates usefull properties like hybridization, ring membership, etc                 
        except:                       #while it checks if the molecuels are reasonable
            print('invalid chemistry')
            return None
    #return both string and molobject
    return oligomer, new_mol


##############################
#ethyleneglycol = OCCO
#1,2-propyleneglycol = OC(C)CO
##############################
monomer = 'OCCO'


#### change filename!!!!
with open('./Polyol_mixtures/csv_files/ethyleneglycol_oligomers_100.csv', 'w',newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['repeatingunits', 'smiles', 'MolWt'])   

    for i in range(1,100):
        Smiles, mol = oligomerGenerator(monomer,i)
        #properties we want to add:
        molWt = ExactMolWt(mol)

        writer.writerow([i, Smiles, molWt])

