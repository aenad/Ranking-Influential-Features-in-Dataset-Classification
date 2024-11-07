# Ranking-Influential-Features-in-Dataset-Classification

## Overview
This project focuses on ranking influential features in dataset classification using unsupervised and supervised learning techniques. The dataset used contains a large number of features, and the project aims to determine which features have the most impact on clustering and classification.

## Project Stages

1. **Data Preprocessing**:  
   - Removed unnecessary columns (`index` and `class label`) from the dataset.
   - Applied **Z-score normalization** to scale all features to a uniform range.

2. **Feature Analysis**:  
   - **Correlation Analysis**: Calculated **Pearson correlation coefficients** to determine feature relationships. Plotted a heatmap to visualize correlations.
   - **Feature Ranking**: Ranked features based on their influence on the dataset.

3. **Unsupervised Learning**:  
   - **K-means Clustering**: Applied K-means clustering with different values of `k`. Used the **Elbow method** to determine the best `k` value. Evaluated clusters using the **Silhouette score**.
   - **Feature Importance in Clustering**: After clustering, features were ranked based on **WCSS (Within-Cluster Sum of Squares)** to determine their impact on clustering cohesion and compactness.

4. **Dimensionality Reduction**:  
   - Used **PCA (Principal Component Analysis)** to reduce the dimensionality of the dataset for visualization and efficiency.
   - Plotted feature importance based on variance to highlight which features contributed most to each cluster.

5. **Supervised Learning**:  
   - Implemented several models for classification, including **Random Forest**, **Gradient Boosting**, and **Decision Tree**.
   - Compared feature importance as determined by each model. **Random Forest** was found to perform best in terms of overall evaluation metrics, although the dataset exhibited class imbalance.

6. **Evaluation and Metrics**:  
   - Evaluated models using metrics such as **ROC**, **Silhouette score**, **Calinski-Harabasz index**, **Davies-Bouldin index**, and **Distortion score**.
   - Conducted **undersampling** to handle class imbalance, improving model performance on less frequent classes.

## Techniques Used
- **Data Preprocessing**: Normalization using Z-score, feature selection, handling missing values.
- **Feature Ranking**: Calculated correlations and evaluated feature importance using both unsupervised and supervised techniques.
- **Unsupervised Learning**: K-means clustering to determine feature influence.
- **Dimensionality Reduction**: Used **PCA** to improve cluster visualization.
- **Supervised Learning**: Compared multiple classification models to rank feature importance.

## Results Summary
- The **K-means clustering** algorithm identified influential features based on cluster cohesion, with feature rankings indicating which features played a significant role in cluster formation.
- **PCA** analysis showed that certain features had consistently high variance, indicating their significance across different clusters.
- **Random Forest** emerged as the best classifier with high accuracy, but class imbalance posed challenges that were mitigated using undersampling.
- Evaluation metrics, including the **Silhouette score** and **Calinski-Harabasz index**, were used to assess clustering quality, while **ROC** curves compared classification models.

## How to Run
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Preprocess Data**: Run the preprocessing script to clean and normalize the dataset.
4. **Train the Model**: Execute the clustering and classification scripts to train the models.
5. **Evaluate and Visualize**: Use the evaluation script to analyze feature importance and visualize the results.

## Dependencies
- Python 3.x
- Libraries: NumPy, pandas, scikit-learn, Matplotlib, Seaborn

## Project Structure
- `notebooks/`: Contains Jupyter notebooks for feature analysis, clustering, and classification.
- `scripts/`: Python scripts for data preprocessing, feature extraction, clustering, and model training.
- `data/`: Directory for storing the dataset.

## License
This project is licensed under the MIT License.

## Acknowledgments
This project was developed as part of a data mining course under the supervision of **Dr. Hossein Rahmani**.
