import pandas as pd
import rdkit as rd
from rdkit import Chem

csv_path = "d:\\ChemData\\tox21.csv"
csv_path = "C:\\Data\\tox21.csv"
tox = pd.read_csv(csv_path)
tox.dropna(inplace=True)
tox.reset_index(drop=True,inplace=True)
tox["mol"] = tox["smiles"].apply(Chem.MolFromSmiles)
from rdkit.Chem import rdFingerprintGenerator
import numpy as np
rd_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
tox["fp"] = tox["mol"].apply(rd_fpgen.GetFingerprint)
#Join all items in a tuple into a string, using a hash character as separator:
#myTuple = ("John", "Peter", "Vicky")
#x = "#".join(myTuple)
# x = John#Peter#Vicky
tox["fp_bits"]=tox["fp"].apply(lambda x:','.join(x.ToBitString()))
data = pd.DataFrame(tox["fp_bits"].str.split(",").tolist(),columns=["fp"+str(x) for x in range(2048)])
#train_data = pd.concat([train_x, tox["NR-AR"]], ignore_index=True, sort=False,axis=1)
target = tox["NR-AR"]

from sklearn.model_selection import train_test_split
train_x, test_x, train_y,test_y = train_test_split(data,target,test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(train_x,train_y)

from sklearn.metrics import f1_score
test_pred = rf.predict(test_x)
print(f1_score(test_y,test_pred))
from sklearn.metrics import roc_auc_score
print(roc_auc_score(test_y,test_pred))

#from sklearn.pipeline import Pipeline
#pipe = Pipeline(("RandonForset",rf))
#pipe.fit(train_x,train_y).score(test_x,test_y)

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = SVC()
clf = GridSearchCV(svc, parameters)
clf.fit(train_x,train_y)
#sorted(clf.cv_results_.keys())
test_pred = clf.predict(test_x)
print(f1_score(test_y,test_pred))
from sklearn.metrics import roc_auc_score
print(roc_auc_score(test_y,test_pred))
