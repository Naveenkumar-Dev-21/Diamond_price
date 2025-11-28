import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np

df = pd.read_csv("gemstone.csv")

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]

df["price_per_carat"] = df["price"] / df["carat"]
df["log_carat"] = np.log1p(df["carat"])
df["depth_ratio"] = df["depth"] / df["table"]

df = pd.get_dummies(df, drop_first=True)

X = df.drop("price", axis=1)
y = df["price"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

tolerance = 0.1
accuracy = np.mean(np.abs(y_test - y_pred) / y_test <= tolerance) * 100

cv_scores = cross_val_score(ridge, X_train, y_train, cv=5, scoring='r2')

print(f"===== Improved Linear Regression Model Performance =====")
print(f"Model: Ridge Regression with Feature Engineering")
print(f"R² Score: {r2:.4f}")
print(f"Cross-Validation R² (avg): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"RMSE: ${rmse:.2f}")
print(f"Accuracy (within 10% tolerance): {accuracy:.2f}%")
print()

# New diamonds (raw)
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

# Convert new data to dummies
new_diamonds = pd.get_dummies(new_diamonds, drop_first=True)

# Add engineered features
new_diamonds["price_per_carat"] = new_diamonds["carat"] / new_diamonds["carat"]
new_diamonds["log_carat"] = np.log1p(new_diamonds["carat"])
new_diamonds["depth_ratio"] = new_diamonds["depth"] / new_diamonds["table"]

# Align columns with training data
new_diamonds = new_diamonds.reindex(columns=X_scaled.columns, fill_value=0)

# Scale the new diamonds data
new_diamonds_scaled = pd.DataFrame(
    scaler.transform(new_diamonds),
    columns=X_scaled.columns
)

# Predict prices for new diamonds
predictions = ridge.predict(new_diamonds_scaled)

print(f"Predicted prices for new diamonds:")
for i, price in enumerate(predictions):
    print(f"Diamond {i+1}: ${price:.2f}")

