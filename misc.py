import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def load_data():
    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw = pd.read_csv(url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw.values[::2, :], raw.values[1::2, :2]])
    target = raw.values[1::2, 2]
    feature_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM",
        "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df["MEDV"] = target
    print("Dataset loaded successfully. Shape:", df.shape)
    return df

def preprocess_data(df):
    X = df.drop("MEDV", axis=1).astype(float)
    y = df["MEDV"].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    #print(f"Training set size: {X_train.shape}, Test set size: {X_test.shape}")
    return X_train, X_test, y_train, y_test

def train_and_evaluate(model, X_test, y_test):
    #print(f"Training model: {model.__class__.__name__} ....")
    #model.fit(X_train, y_train)
    print("Model training complete. Evaluating on test set...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Evaluation complete. MSE: {mse:.4f}")
    return mse

def run_training_pipeline(model_class):
    print("Loading data...")
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = model_class()
    print(f"Training model: {model.__class__.__name__} ...")
    model.fit(X_train, y_train)
    mse = train_and_evaluate(model, X_test, y_test)
    print(f"Final Test MSE: {mse:.4f}")
    return model, (X_train, X_test, y_train, y_test), mse

