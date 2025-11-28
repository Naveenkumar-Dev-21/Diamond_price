import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


df = pd.read_csv("gemstone.csv")


df["price_per_carat"] = df["price"] / df["carat"]

df = pd.get_dummies(df)


X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Linear Regression Model Performance:")
print(f"RMSE: ${rmse:.2f}")
print(f"R² Score: {r2:.4f}")
print()


new_diamonds = pd.DataFrame({
    'carat': [0.23, 0.57],
    'cut': ['Premium', 'Ideal'],
    'color': ['E', 'I'],
    'clarity': ['VS2', 'VS1'],
    'depth': [61.5, 62.2],
    'table': [55, 58],
    'x': [3.95, 5.73],
    'y': [3.98, 5.75],
    'z': [2.43, 3.57]
})

new_diamonds = pd.get_dummies(new_diamonds)

new_diamonds = new_diamonds.reindex(columns=X.columns, fill_value=0)


predictions = lr.predict(new_diamonds)

print(f"Predicted prices for new diamonds:")
for i, price in enumerate(predictions):
    print(f"Diamond {i+1}: ${price:.2f}")

