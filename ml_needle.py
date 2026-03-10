import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from joblib import dump


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


def extract_polymer_features(row):

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
        df.apply(extract_polymer_features, axis=1)
    )

    categorical = pd.get_dummies(
        df[["GMT","MBR"]]
    ).values

    numeric_cols = df.drop(
        columns=[
            "SMILES","poli1","poli2","poli3",
            "GMT","MBR",
            "JPR","JPR_SD","PPR"
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

def svm_cross_validation(X_train, y_train, folds=5):

    model = make_pipeline(
        StandardScaler(),
        SVR()
    )

    kf = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring="neg_mean_squared_error"
    )

    rmse_scores = np.sqrt(-scores)

    print("RMSE per fold:", rmse_scores)
    print("Mean RMSE:", rmse_scores.mean())
    print()

    return rmse_scores

def svm_cv_tuned(X_train, y_train):

    pipeline = make_pipeline(
        StandardScaler(),
        SVR()
    )

    param_grid = [

        # RBF kernel
        {
            "svr__kernel": ["rbf"],
            "svr__C": [0.1, 1, 10, 100, 1000],
            "svr__gamma": ["scale", 0.001, 0.01, 0.1, 1],
            "svr__epsilon": [0.001, 0.01, 0.1, 0.5]
        },

        # Linear kernel
        {
            "svr__kernel": ["linear"],
            "svr__C": [0.1, 1, 10, 100, 1000],
            "svr__epsilon": [0.001, 0.01, 0.1, 0.5]
        },

        # Polynomial kernel
        {
            "svr__kernel": ["poly"],
            "svr__degree": [2, 3, 4],
            "svr__C": [0.1, 1, 10, 100],
            "svr__gamma": ["scale", 0.01, 0.1],
            "svr__epsilon": [0.01, 0.1]
        }
    ]

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    rmse = np.sqrt(-grid.best_score_)

    print("Best params:", grid.best_params_)
    print("Best kernel:", grid.best_params_["svr__kernel"])
    print("CV RMSE:", rmse)

    return grid, rmse

def plot_prediction(y_true, y_pred, title, mode):

    plt.figure(figsize=(8,8))

    plt.scatter(y_true, y_pred, alpha=0.8)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    plt.plot([min_val, max_val], [min_val, max_val], '--')

    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"{title} {mode} Prediction vs True")

    plt.show()


def random_forest_cv(X_train, y_train, n_iter=100):

    pipeline = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(random_state=42)
    )

    param_dist = {
        "randomforestregressor__n_estimators": [100, 200, 500, 800, 1000],
        "randomforestregressor__max_depth": [None, 5, 10, 20, 40],
        "randomforestregressor__min_samples_split": [2, 5, 10],
        "randomforestregressor__min_samples_leaf": [1, 2, 4],
        "randomforestregressor__max_features": ["sqrt", "log2", None],
        "randomforestregressor__bootstrap": [True, False]
    }

    random_search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    rmse = np.sqrt(-random_search.best_score_)

    print("Best params:", random_search.best_params_)
    print("CV RMSE:", rmse)

    return random_search, rmse


def xgboost_cv(X_train, y_train, n_iter=100):

    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1
    )

    param_dist = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [3, 4, 5, 6, 8, 10],
        "learning_rate": [0.001, 0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "gamma": [0, 0.1, 0.2, 0.5],
        "reg_alpha": [0, 0.01, 0.1, 1],
        "reg_lambda": [1, 2, 5]
    }

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)

    rmse = np.sqrt(-random_search.best_score_)

    print("Best params:", random_search.best_params_)
    print("CV RMSE:", rmse)

    return random_search, rmse

def linear_models_cv(X_train, y_train):

    pipeline = make_pipeline(
        StandardScaler(),
        LinearRegression()
    )

    param_grid = [

        # Ordinary Multiple Linear Regression
        {
            "linearregression": [LinearRegression()]
        },

        # Ridge Regression
        {
            "linearregression": [Ridge()],
            "linearregression__alpha": [0.001, 0.01, 0.1, 1, 10, 100]
        },

        # Lasso Regression
        {
            "linearregression": [Lasso(max_iter=10000)],
            "linearregression__alpha": [0.001, 0.01, 0.1, 1, 10]
        },

        # ElasticNet
        {
            "linearregression": [ElasticNet(max_iter=10000)],
            "linearregression__alpha": [0.001, 0.01, 0.1, 1, 10],
            "linearregression__l1_ratio": [0.1, 0.5, 0.9]
        }
    ]

    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    rmse = np.sqrt(-grid.best_score_)

    print("Best model:", type(grid.best_estimator_.named_steps['linearregression']).__name__)
    print("Best params:", grid.best_params_)
    print("CV RMSE:", rmse)

    return grid, rmse

def generate_and_save_folds(df, n_splits=5, random_state=42, output_dir="cv_folds", excel_filename="polymer_cv_folds.xlsx"):
    os.makedirs(output_dir, exist_ok=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    excel_path = os.path.join(output_dir, excel_filename)
    writer = pd.ExcelWriter(excel_path, engine="openpyxl")
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df), start=1):
        fold_train = df.iloc[train_idx].reset_index(drop=True)
        fold_val = df.iloc[val_idx].reset_index(drop=True)
        fold_key = f"fold_{fold_idx}"
        fold_train.to_excel(writer, sheet_name=f"{fold_key}_train", index=False)
        fold_val.to_excel(writer, sheet_name=f"{fold_key}_val", index=False)
        

if __name__=="__main__":
    input_path = r"D:\skripsi_oneng\ml_mn_cur.xlsx"
    output_dir = r"D:\skripsi_oneng\ml_mn_cur_akhir.xlsx"
    os.makedirs(output_dir, exist_ok=True)

    algorithm = "MLR"
    df  = pd.read_excel(input_path)
    initial_count = len(df)
    print(f"Total data mentah: {initial_count} baris.")

    df = df.drop(columns=["ZA","REF"],errors="ignore")
    
    X = build_features(df)
    y = df["PPR"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.2, 
        random_state=42
    )

    grid, cv_rmse = linear_models_cv(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred_train = best_model.predict(X_train)
    plot_prediction(y_train, y_pred_train, title = algorithm, mode = "train")

    y_pred_test = best_model.predict(X_test)
    plot_prediction(y_test, y_pred_test, title = algorithm, mode = "test")

    df_test = pd.DataFrame({
    "y_test": y_test,
    "y_pred_test": y_pred_test
        })

    df_test.to_excel(f"{algorithm}_pred_test.xlsx", index=False)

    df_train = pd.DataFrame({
    "y_train": y_train,
    "y_pred_train": y_pred_train
        })

    df_train.to_excel(f"{algorithm}_pred_train.xlsx", index=False)

    dump(grid, f"{algorithm}_grid_search.pkl")

    dump(grid.best_estimator_, f"{algorithm}_best_model.pkl")
    print("test")
