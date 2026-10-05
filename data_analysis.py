import pandas as pd

data = {
    "transaction_id": ["T001", "T002", "T003", "T004", "T005", "T006", "T007"],
    "region": ["South", "North", "South", "West", "North", "South", "West"],
    "product": ["Laptop", "Monitor", "Server", "Laptop", "Monitor", "Laptop", "Server"],
    "quantity": [10, 5, 2, 8, 6, 12, 1],
    "revenue": [100000, 120000, 95000, 110000, 105000, 130000, 50000000]
}

df = pd.DataFrame(data)
print(df)
print(df["revenue"].describe())
Q1 = df["revenue"].quantile(0.25)
Q3 = df["revenue"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)
outliers = df[
    (df["revenue"] < lower_limit) |
    (df["revenue"] > upper_limit)
]

print(outliers)
print(df.groupby("region")["revenue"].sum())
print(df.groupby("product")["revenue"].sum())
print(df.groupby("product")["quantity"].sum())
df["revenue_per_unit"] = df["revenue"] / df["quantity"]

print(df[["product", "quantity", "revenue", "revenue_per_unit"]])
print(df[["quantity", "revenue"]].corr())
print(df.dtypes)
print(df.duplicated().sum())
print(df.isnull().sum())

X = df[["region", "product", "quantity"]]
y = df["revenue"]
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)

X_encoded = encoder.fit_transform(X[["region", "product"]])
print(X_encoded)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(), ["region", "product"])
    ],
    remainder="passthrough"
)
X_processed = preprocessor.fit_transform(X)

print(X_processed)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.2,
    random_state=42
)
print("X_train:")
print(X_train)

print("X_test:")
print(X_test)

print("y_train:")
print(y_train)

print("y_test:")
print(y_test)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)
print("Model trained successfully")
y_pred = model.predict(X_test)

print("Actual values:")
print(y_test)

print("\nPredicted values:")
print(y_pred)
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print("MAE:", mae)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)

print("MSE:", mse)
import numpy as np

rmse = np.sqrt(mse)

print("RMSE:", rmse)
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("R²:", r2)
print(df[df["revenue"] > 1000000])
df_clean = df[df["revenue"] <= 1000000].copy()

print(df_clean)
X = df_clean[["region", "product", "quantity"]]
y = df_clean["revenue"]

print(X)
print(y)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("X_train:")
print(X_train)

print("\nX_test:")
print(X_test)

print("\ny_train:")
print(y_train)

print("\ny_test:")
print(y_test)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["region", "product"])
    ],
    remainder="passthrough"
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("X_train_processed:")
print(X_train_processed)

print("\nX_test_processed:")
print(X_test_processed)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train_processed, y_train)

print("Model trained successfully")
y_pred = model.predict(X_test_processed)

print("Actual values:")
print(y_test)

print("\nPredicted values:")
print(y_pred)
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)

print("MAE:", mae)
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("MSE:", mse)
print("RMSE:", rmse)
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print("R²:", r2)
from sklearn.tree import DecisionTreeRegressor

tree_model = DecisionTreeRegressor(random_state=42)

tree_model.fit(X_train_processed, y_train)

print("Decision Tree trained successfully")
tree_pred = tree_model.predict(X_test_processed)

print("Actual values:")
print(y_test)

print("\nDecision Tree predictions:")
print(tree_pred)
tree_mae = mean_absolute_error(y_test, tree_pred)

print("Decision Tree MAE:", tree_mae)
deep_tree = DecisionTreeRegressor(
    max_depth=None,
    random_state=42
)

deep_tree.fit(X_train_processed, y_train)

train_pred = deep_tree.predict(X_train_processed)

train_mae = mean_absolute_error(y_train, train_pred)

print("Training MAE:", train_mae)
controlled_tree = DecisionTreeRegressor(
    max_depth=2,
    random_state=42
)

controlled_tree.fit(X_train_processed, y_train)

train_pred = controlled_tree.predict(X_train_processed)
test_pred = controlled_tree.predict(X_test_processed)

train_mae = mean_absolute_error(y_train, train_pred)
test_mae = mean_absolute_error(y_test, test_pred)

print("Training MAE:", train_mae)
print("Test MAE:", test_mae)
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

forest_model.fit(X_train_processed, y_train)

print("Random Forest trained successfully")
forest_pred = forest_model.predict(X_test_processed)

print("Actual values:")
print(y_test)

print("\nRandom Forest predictions:")
print(forest_pred)
forest_mae = mean_absolute_error(y_test, forest_pred)

print("Random Forest MAE:", forest_mae)
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(
    forest_model,
    X_train_processed,
    y_train,
    cv=3,
    scoring="neg_mean_absolute_error"
)

print(cv_scores)
cv_mae = -cv_scores.mean()

print("Cross-validation MAE:", cv_mae)
from sklearn.ensemble import RandomForestRegressor

for n in [10, 50, 100, 200]:
    model = RandomForestRegressor(
        n_estimators=n,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X_train_processed,
        y_train,
        cv=3,
        scoring="neg_mean_absolute_error"
    )

    mae = -scores.mean()

    print("Trees:", n, "CV MAE:", mae)
for depth in [2, 3, 5, None]:
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=depth,
        random_state=42
    )

    scores = cross_val_score(
        model,
        X_train_processed,
        y_train,
        cv=3,
        scoring="neg_mean_absolute_error"
    )

    mae = -scores.mean()

    print("Max depth:", depth, "CV MAE:", mae)
    feature_names = preprocessor.get_feature_names_out()

print("Feature names:")
print(feature_names)
importances = forest_model.feature_importances_

for name, importance in zip(feature_names, importances):
    print(name, ":", importance)
    new_transaction = pd.DataFrame({
    "region": ["South"],
    "product": ["Laptop"],
    "quantity": [15]
})

print(new_transaction)
new_transaction_processed = preprocessor.transform(new_transaction)

print(new_transaction_processed)
new_prediction = forest_model.predict(new_transaction_processed)

print("Predicted revenue:", new_prediction[0])
from sklearn.ensemble import IsolationForest

revenue_data = df[["revenue"]]

print(revenue_data)
anomaly_model = IsolationForest(
    contamination=0.15,
    random_state=42
)
anomaly_model = IsolationForest(
    contamination=0.15,
    random_state=42
)
anomaly_model.fit(revenue_data)
anomaly_predictions = anomaly_model.predict(revenue_data)

print("Anomaly predictions:")
print(anomaly_predictions)
anomaly_scores = anomaly_model.decision_function(revenue_data)

print("Anomaly scores:")
print(anomaly_scores)
import numpy as np

inputs = np.array([2, 3])
weights = np.array([0.5, 0.8])
bias = 1

output = np.dot(inputs, weights) + bias

print(output)
def relu(x):
    return max(0, x)

print(relu(4.4))
print(relu(-3.2))
actual = 10
prediction = 8

loss = (actual - prediction) ** 2

print(loss)
data = np.arange(10)

batch_size = 2
epochs = 3

for epoch in range(epochs):
    print("Epoch:", epoch + 1)

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        print("  Batch:", batch)
import torch

X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[2.0], [4.0], [6.0], [8.0], [10.0]])

print(X)
print(y)
import torch.nn as nn

model = nn.Linear(1, 1)

print(model)
print("Weight:", model.weight)
print("Bias:", model.bias)
predictions = model(X)

print("Predictions:")
print(predictions)
loss_function = nn.MSELoss()

loss = loss_function(predictions, y)

print("Loss:", loss.item())
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(1000):
    predictions = model(X)
    loss = loss_function(predictions, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("Weight:", model.weight.item())
print("Bias:", model.bias.item())
predictions = model(X)

print("Final predictions:")
print(predictions)