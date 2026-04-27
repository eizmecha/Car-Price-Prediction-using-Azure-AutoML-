# 🚗 Automobile Price Prediction: End-to-End ML Pipeline on Azure

## 📌 Executive Summary
This project demonstrates an industrial-grade Machine Learning workflow to predict automobile prices. Leveraging **Microsoft Azure Machine Learning (AutoML)**, I developed a high-precision regression model that analyzes mechanical specifications (Engine size, Horsepower, Curb weight) to estimate market value.

## 🏗️ System Architecture & Workflow
The project follows the standard CRISP-DM methodology implemented on the Cloud:
1. **Data Ingestion:** Registering raw datasets in Azure ML Data Assets.
2. **Preprocessing:** Handling missing values (represented by "?" in raw data) and feature scaling.
3. **Automated ML:** Running parallel experiments to find the optimal regression algorithm.
4. **Model Evaluation:** Analyzing Residuals, MAE, and R-Squared metrics.
5. **Registration:** Deploying the best model as a cloud-hosted endpoint.



## 📊 Dataset Insights
* **Source:** UCI Machine Learning Repository (Automobile Dataset).
* **Key Features:** `engine-size`, `horsepower`, `width`, `curb-weight`.
* **Target Variable:** `price` (Regression Task).

## 🤖 Model Performance
The **VotingEnsemble** algorithm outperformed other candidates:
* **R2 Score:** `0.895` (The model explains ~90% of the price variance).
* **Normalized RMSE:** `0.0495`.

### 📈 Results Visualization
![Model Evaluation](Visuals/Results.png)
*The plot illustrates the strong alignment between predicted values and actual market prices.*

## 🛠️ Technology Stack
* **Cloud Platform:** Microsoft Azure (Azure ML Studio).
* **Tools:** AutoML, Designer, Azure Compute Clusters.
* **Environment:** Python, Jupyter Notebooks.
* **Frameworks:** Scikit-learn, Pandas, NumPy.

## 🚀 Deployment
The model is registered and ready for production. It can be consumed via a **REST API Endpoint** for real-time inference in automotive diagnostic apps.
