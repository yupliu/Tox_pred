import pandas as pd
import rdkit as rd
from rdkit import Chem

csv_path = "d:\\ChemData\\tox21.csv"
tox = pd.read_csv(csv_path)
tox.dropna(inplace=True)
tox["mol"] = tox["smiles"].apply(Chem.MolFromSmiles)
