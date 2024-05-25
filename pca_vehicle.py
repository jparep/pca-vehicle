# Import necessary libraries
import pandas as pd  # For data manipulation and analysis
from sklearn.preprocessing import StandardScaler  # For standardizing the features
from sklearn.decomposition import PCA  # For performing Principal Component Analysis
import matplotlib.pyplot as plt  # For data visualization

# Step 1: Load the Data
# Load the dataset from the URL
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/vehicle.dat"
data = pd.read_csv(url, delimiter='\\s+', header=None)

# Display the first few rows of the dataset to understand its structure
print("First few rows of the dataset:")
print(data.head())

# Step 2: Data Preprocessing
# Separate features and labels
X = data.iloc[:, :-1]  # Features (all columns except the last one)
y = data.iloc[:, -1]   # Labels (the last column)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Apply PCA
# Apply PCA and reduce the data to 2 principal components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Check the explained variance ratio of the principal components
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by each component: {explained_variance}")

# Step 4: Visualize the Results
# Create a DataFrame with the PCA results and the corresponding labels
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['Label'] = y

# Plot the PCA results
plt.figure(figsize=(8, 6))
for label in pca_df['Label'].unique():
    subset = pca_df[pca_df['Label'] == label]
    plt.scatter(subset['PC1'], subset['PC2'], label=label)

# Add plot labels and title
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Vehicle Dataset')
plt.legend()

# Show the plot
plt.show()
