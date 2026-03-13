import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.stats as stats

from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from joblib import dump, load
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error



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


def extract_polymer_features(row, radius, n_bits):

    fps = []

    for col in ["poli1","poli2","poli3"]:

        smiles = row[col]

        if pd.isna(smiles):
            fps.append(np.zeros(n_bits))
        else:
            fps.append(smiles_to_ecfp(smiles, radius = radius, n_bits=n_bits))

    return np.concatenate(fps)

def build_features(
    df,
    radius=2,
    n_bits=2048,
    use_smiles=True,
    use_polymer=False,
    use_categorical=True,
    use_numeric=True
):

    features = []

    if use_smiles and "SMILES" in df.columns:

        ligand_features = np.vstack(
            df["SMILES"].apply(smiles_to_ecfp, args=(radius, n_bits))
        )

        features.append(ligand_features)

    if use_polymer:

        polymer_features = np.vstack(
            df.apply(extract_polymer_features, args=(radius, n_bits), axis=1)
        )

        features.append(polymer_features)

    if use_categorical:
        cat_cols = [c for c in ["GMT", "MBR"] if c in df.columns]

        if len(cat_cols) > 0:
            categorical = pd.get_dummies(df[cat_cols]).values

            features.append(categorical)

    if use_numeric:
        drop_cols = [
            "SMILES","poli1","poli2","poli3",
            "GMT","MBR",
            "JPR","JPR_SD","PPR"
        ]

        numeric_cols = df.drop(
            columns=[c for c in drop_cols if c in df.columns]
        )

        if numeric_cols.shape[1] > 0:
            numeric = numeric_cols.values

            features.append(numeric)

    X = np.concatenate(features, axis=1)

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
        "randomforestregressor__n_estimators": [100, 200, 500, 800, 100],
        "randomforestregressor__max_depth": [None, 20, 40],
        "randomforestregressor__min_samples_split": [2, 5, 10],
        "randomforestregressor__min_samples_leaf": [1, 2, 4],
        "randomforestregressor__max_features": ["sqrt", "log2"]
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
        "max_depth": [2, 4, 6, 8, 16, 32],
        "min_child_weight": [1, 5, 10],
        "subsample": [0.4, 0.6, 1.0],
        "colsample_bytree": [0.5, 0.8, 1.0],
        "gamma": [0, 0.1, 0.5]
        
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
    
    radius = 2
    n_bits = 2048
    algorithm = "MLR"

    df  = pd.read_excel(input_path)
    initial_count = len(df)
    print(f"Total data mentah: {initial_count} baris.")

    df = df.drop(columns=["ZA","REF"],errors="ignore")
    
    X = build_features(df, radius=radius, n_bits=n_bits, use_polymer = False)
    y = df["PPR"].values


    if 1: # pengulangan (default 5 kali)
        grid = load(f"{algorithm}_r{radius}n{n_bits}_PPR_best_model.pkl")        
        n_repeats = 5
        
        metrics_list = []

        best_params = grid[1]

        for i in range(n_repeats):
        
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=42 + i
            )
        
            # rebuild model dengan hyperparameter terbaik
            if algorithm == "MLR":
                model = grid[1]
        
            elif algorithm == "RF":
                model = RandomForestRegressor(
                    **{k.split("__")[-1]: v for k,v in best_params.items()},
                    random_state=42
                )
        
            elif algorithm == "XGB":
                model = XGBRegressor(
                    **best_params,
                    objective="reg:squarederror",
                    random_state=42
                )
        
            model.fit(X_train, y_train)
        
            y_pred_test = model.predict(X_test)
        
            r2 = r2_score(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            mae = mean_absolute_error(y_test, y_pred_test)
            spearman, pvalue = stats.spearmanr(y_test, y_pred_test)
        
            metrics_list.append({
                "repeat": i+1,
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "Spearman": spearman
            })
        
        metrics_df = pd.DataFrame(metrics_list)
        
        summary = metrics_df.mean(numeric_only=True)
        
        print(metrics_df)
        print("\nAverage metrics:")
        print(summary)

        metrics_df.to_excel(
        f"{algorithm}_r{radius}n{n_bits}_PPR_repeated_test_metrics.xlsx",
        index=False
        )

    if 0: # 1 kali
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.3, 
            random_state=42
        )
    
        if algorithm == "MLR":
            grid, cv_rmse = linear_models_cv(X_train, y_train)
        
        elif algorithm == "RF":
            grid, cv_rmse = random_forest_cv(X_train, y_train, n_iter=100)
        
        elif algorithm == "XGB":
            grid, cv_rmse = xgboost_cv(X_train, y_train, n_iter=100)
        
        else:
            raise ValueError("Algorithm not recognized")
    
        best_model = grid.best_estimator_
    
        y_pred_train = best_model.predict(X_train)
        plot_prediction(y_train, y_pred_train, title = algorithm, mode = "train")
    
        y_pred_test = best_model.predict(X_test)
        plot_prediction(y_test, y_pred_test, title = algorithm, mode = "test")
    
        df_test = pd.DataFrame({
        "y_test": y_test,
        "y_pred_test": y_pred_test
            })
    
        df_test.to_excel(f"{algorithm}_r{radius}n{n_bits}_pred_PPR_test.xlsx", index=False)
    
        df_train = pd.DataFrame({
        "y_train": y_train,
        "y_pred_train": y_pred_train
            })
    
        df_train.to_excel(f"{algorithm}_r{radius}n{n_bits}_pred_PPR_train.xlsx", index=False)
    
        dump(grid, f"{algorithm}_r{radius}n{n_bits}_PPR_grid_search.pkl")
    
        dump(grid.best_estimator_, f"{algorithm}_r{radius}n{n_bits}_PPR_best_model.pkl")
    
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
    
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
    
        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)
    
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
    
        spearman_train, pvalue_train = stats.spearmanr(y_train, y_pred_train)
        spearman_test, pvalue_test = stats.spearmanr(y_test, y_pred_test)
    
        metrics_df = pd.DataFrame({
        "Dataset": ["Train", "Test"],
        "R2": [r2_train, r2_test],
        "RMSE": [rmse_train, rmse_test],
        "MAE": [mae_train, mae_test],
        "Spearman": [spearman_train, spearman_test],
        "Spearman_pvalue": [pvalue_train, pvalue_test]
        })
    
        metrics_df.to_excel(
        f"{algorithm}_r{radius}n{n_bits}_PPR_metrics.xlsx",
        index=False
        )
    
        print(metrics_df)
