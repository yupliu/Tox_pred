import pandas as pd
import rdkit as rd
from rdkit import Chem

csv_path = "d:\\ChemData\\tox21.csv"
csv_path = "C:\\Data\\tox21.csv"
tox = pd.read_csv(csv_path)
tox.dropna(inplace=True)
tox["mol"] = tox["smiles"].apply(Chem.MolFromSmiles)


from rdkit.Chem import rdFingerprintGenerator
import numpy as np
rd_fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
tox["fp"] = tox["mol"].apply(rd_fpgen.GetFingerprint)
tox["fp_bits"]=tox["fp"].apply(lambda x:x.ToBitString())
