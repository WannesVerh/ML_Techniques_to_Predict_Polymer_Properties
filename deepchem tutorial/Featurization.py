import deepchem as dc
from rdkit.Chem import Descriptors

print("######################################################################################",'\n')
featurizer = dc.feat.CircularFingerprint()
print(featurizer(['CC', 'CCC', 'CCO']))

'''RDKITfeaturizer = dc.feat.RDKitDescriptors()
print(RDKITfeaturizer.descpriptors)
features = RDKITfeaturizer(['CC'])[0]
for decriptornr, value in enumerate(features):
    print(RDKITfeaturizer.descriptors[decriptornr], " : ", value)'''

rdkit_featurizer = dc.feat.RDKitDescriptors()
features = rdkit_featurizer.featurize(['CCC'])
print(features[0])
