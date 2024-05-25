# Dimensionality Reduction and Visualization of Vehicle Data using Principal Component Analysis (PCA)

## Overview
This project demonstrates the use of Principal Component Analysis (PCA) to reduce the dimensionality of a dataset containing physical measurements of different types of vehicles. The goal is to visualize the high-dimensional data in a 2D space and understand the underlying structure of the data.

## Dataset
The dataset used in this project is the Vehicle dataset from the UCI Machine Learning Repository. It contains various physical measurements of different types of vehicles.

- **URL:** [Vehicle Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/vehicle/vehicle.dat)

## Project Steps

1. **Load the Data:**
   - Load the dataset from the URL.
   - Display the first few rows of the dataset to understand its structure.

2. **Data Preprocessing:**
   - Separate features and labels.
   - Standardize the features to have zero mean and unit variance.

3. **Apply PCA:**
   - Apply PCA to reduce the data to 2 principal components.
   - Check the explained variance ratio of the principal components.

4. **Visualize the Results:**
   - Create a DataFrame with the PCA results and the corresponding labels.
   - Plot the PCA results in a 2D scatter plot, color-coded by the labels.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/jparep/pca-vehicle.git
    cd pca-vehicle
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    conda env create -f environment.yml
    conda activate pcaenv
    ```

3. Install the required packages:
    ```bash
    pip install pandas scikit-learn matplotlib
    ```

## Usage

1. Run the script:
    ```bash
    python pca_vehicle.py
    ```

2. The script will load the dataset, preprocess the data, apply PCA, and visualize the results.

## Example Output
The script will produce a 2D scatter plot showing the first two principal components of the vehicle data, with points color-coded by their labels (types of vehicles).

![PCA Visualization](example_plot.png)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements
- UCI Machine Learning Repository for providing the Vehicle dataset.
- [scikit-learn](https://scikit-learn.org/stable/) for the PCA implementation.
- [matplotlib](https://matplotlib.org/) for the visualization tools.
