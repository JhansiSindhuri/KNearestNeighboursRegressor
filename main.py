import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error

# Read the CSV file from the local path
csv_path = "C:\\Users\\LENOVO\\Documents\\Python Projects\\House Project\\house.csv"
data = pd.read_csv(csv_path)

# Continue with the rest of your code...

# Step 2: Exploratory Data Analysis
# Display summary statistics
print(data.describe())

# Visualizations (customize as needed)
plt.figure(figsize=(12, 6))
sns.histplot(data['Price'], kde=True)
plt.xlabel('Price')
plt.title('Distribution of House Prices')
plt.show()

sns.pairplot(data[['Sqft', 'Floor', 'TotalFloor', 'Bedroom', 'Living.Room', 'Bathroom', 'Price']])
plt.show()

# Step 3: Data Splitting
X = data[['Sqft', 'Floor', 'TotalFloor', 'Bedroom', 'Living.Room', 'Bathroom']]
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: K-Nearest Neighbors Model
k_values = [1, 3, 5, 7, 9]
best_k = None
best_mae = float('inf')

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MAE for k={k}: {mae}")
    
    if mae < best_mae:
        best_mae = mae
        best_k = k

print(f"Best k: {best_k}, Lowest MAE: {best_mae}")

# Step 5: Train the final model using the best k value and make predictions
final_knn = KNeighborsRegressor(n_neighbors=best_k)
final_knn.fit(X_train, y_train)
y_pred_final = final_knn.predict(X_test)
