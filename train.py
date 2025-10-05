from misc import train_and_evaluate, load_data, preprocess_data
from sklearn.tree import DecisionTreeRegressor


df = load_data()
X_train, X_test, y_train, y_test = preprocess_data(df)
model = DecisionTreeRegressor()
mse = train_and_evaluate(model, X_train, X_test, y_train, y_test)
print(f"DecisionTreeRegressor Test MSE: {mse:.2f}")
