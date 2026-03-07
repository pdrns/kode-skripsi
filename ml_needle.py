import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR



def smiles_to_ecfp(smiles, radius=2, n_bits=2048):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return np.zeros(n_bits)

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=n_bits
    )

    return np.array(fp)


def polymer_features(row):

    fps = []

    for col in ["poli1","poli2","poli3"]:

        smiles = row[col]

        if pd.isna(smiles):
            fps.append(np.zeros(1024))
        else:
            fps.append(smiles_to_ecfp(smiles, n_bits=1024))

    return np.concatenate(fps)

def build_features(df):

    ligand_features = np.vstack(
        df["SMILES"].apply(smiles_to_ecfp)
    )

    polymer_features = np.vstack(
        df.apply(polymer_features, axis=1)
    )

    categorical = pd.get_dummies(
        df[["GMT","MBR"]]
    ).values

    numeric_cols = df.drop(
        columns=[
            "SMILES","poli1","poli2","poli3",
            "GMT","MBR",
            "JPR","JPR_SD"
        ]
    )

    numeric = numeric_cols.values

    X = np.concatenate([
        ligand_features,
        polymer_features,
        categorical,
        numeric
    ], axis=1)

    return X


model = RandomForestRegressor(
    n_estimators=500,
    random_state=42
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVR())
])
