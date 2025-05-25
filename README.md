# Dynamic Fashion Pricing API 

This project presents a complete end-to-end solution for **Dynamic Pricing of fashion items**. It showcases the full machine learning project lifecycle: from initial data exploration and feature engineering, through rigorous model training and evaluation, to the development and containerization of a predictive API.

##  Overview

The Dynamic Fashion Pricing API offers an intelligent system for fashion retailers to optimize pricing strategies. By leveraging machine learning, it provides real-time price recommendations. The core of this project lies in a robust predictive model, developed after thorough data analysis, feature engineering, and comparative evaluation of multiple regression algorithms. The final outcome is a containerized FastAPI application serving these predictions.


## Key Stages & Components

This project encompasses the following completed stages:

1.  **Exploratory Data Analysis (EDA)**:
    * Check out [EDA](https://www.kaggle.com/code/mehrajalomtapadar/eda01) for NOTEBOOK.
    * Comprehensive analysis of the fashion dataset to identify trends, distributions, and correlations relevant to pricing.
    * Data cleaning and visualization to uncover key insights.
    * (Primarily conducted in Jupyter Notebooks like `notebooks/eda01 (4).ipynb`)

2.  **Feature Engineering (FE)**:
    * Check out [Feature Engineering](https://www.kaggle.com/code/mehrajalomtapadar/fe-01) for NOTEBOOK.
    * Transformation of raw data into meaningful features to enhance model predictive power. Key features engineered and used include `BrandName`, `Category`, `MRP`, `Details` (text-based), and `Sizes_str` (processed from available sizes).
    * Techniques included cleaning and standardizing categorical and text data.
    * (Insights and processes developed in Jupyter Notebooks like `notebooks/fe-01 (1).ipynb`)

3.  **Model Development & Evaluation**:
    * **Preprocessing**: A pipeline was constructed using `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer` involving:
        * `OneHotEncoder` for categorical features (`BrandName`, `Category`).
        * `StandardScaler` for numerical features (`MRP`).
        * `TfidfVectorizer` for text features (`Details`, `Sizes_str`).
        * `PCA` (Principal Component Analysis) for dimensionality reduction (retaining 95% of variance or a fixed number of components).
    * **Model Training**: Three regression models were trained and evaluated:
        * XGBoost Regressor (`XGBRegressor`)
        * Random Forest Regressor (`RandomForestRegressor`)
        * LightGBM Regressor (`LGBMRegressor`)
    * **Hyperparameter Tuning**: `GridSearchCV` was used with 3-fold cross-validation to find the best hyperparameters for each model based on `neg_mean_squared_error`.
    * **Evaluation Metrics**: Models were evaluated using Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R-squared (R²).
    * **Model Persistence**: The best performing pipeline (including preprocessor, PCA, and model) for each algorithm was saved using `joblib` to `.pkl` files in the `models/` directory.

4.  **API Development & Containerization**:
    * A RESTful API was developed using **FastAPI** to serve predictions from the trained models. (The API would load one of the saved `.pkl` models).
    * The application, including the model and dependencies, was containerized using **Docker** for portability, scalability, and ease of deployment.


## API Features

* **Dynamic Price Prediction**: Core endpoint (e.g., `/predict`) to serve optimized price predictions for fashion items based on their features.
* **Scalable Architecture**: Built with Python and FastAPI, designed for efficient request handling.
* **Containerized & Deployable**: Dockerized application for consistent environments and straightforward deployment.
* **Machine Learning Powered**: Leverages robust regression models (XGBoost, Random Forest, LightGBM) trained on historical data.


## Technology Stack

* **Backend API**: Python, FastAPI
* **Data Handling & Analysis**: Pandas, NumPy
* **Machine Learning & Preprocessing**: Scikit-learn (Pipeline, ColumnTransformer, OneHotEncoder, StandardScaler, TfidfVectorizer, PCA, train_test_split, GridSearchCV, metrics)
* **ML Models**: XGBoost, RandomForestRegressor, LightGBM
* **Model Persistence**: Joblib
* **Development Environment**: Jupyter Notebooks (for EDA, FE, initial model experimentation),VS CODE
* **Containerization**: Docker
* **Version Control**: Git, GitHub


##  Model Performance Highlights

All trained models demonstrated strong predictive performance on the test set:

* **XGBoost Regressor**:
    * R²: ~0.90
    * MSE: ~48220.20, RMSE: ~219.59, MAE: ~149.55
* **Random Forest Regressor**:
    * R²: ~0.90
    * MSE: ~45949.27, RMSE: ~214.36, MAE: ~138.53
* **LightGBM Regressor**:
    * R²: ~0.90
    * MSE: ~48198.31, RMSE: ~219.54, MAE: ~153.30

An **R² score of approximately 0.90** indicates that the models can explain about 90% of the variance in selling prices, suggesting a good fit to the data. The Random Forest model showed slightly lower error metrics in the reported run.


##  Setup and Usage (API Deployment)

### Prerequisites

* Docker installed on your system.
* Git client (for cloning).

### Running the API Locally with Docker

1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone <your-repository-url>
    cd dynamic_pricing_engine
    ```

2.  **Ensure a trained model (e.g., `rf_model.pkl` or `lgbm_model.pkl`) is present in the `models/` directory**, and your FastAPI application (`app/main.py`) is set up to load it.

3.  **Build the Docker image:**
    (Ensure your `Dockerfile` correctly copies the `app/` directory, `models/` directory, and other necessary files)
    ```bash
    docker build -t fashion-api .
    ```

4.  **Run the Docker container:**
    ```bash
    docker run -d -p 8000:8000 --name fashion-app fashion-api
    ```

5.  **Access the API:**
    * **API Docs (Swagger UI)**: Open your browser and go to `http://localhost:8000/docs`
    * **Prediction Endpoint**: `POST` requests to `http://localhost:8000/predict`
        * **Request Body Example** (adjust based on your `schemas.py` and expected input features):
            ```json
          {
             "BrandName": "Nike",
             "Category": "WesrernWear",
             "MRP": 1999,
             "Details": "cotton summer wear",
             "Sizes": "S"
         }
            ```

---
##  Future Scope & Iterations

* **Refined Hyperparameter Tuning**: Utilize more extensive search spaces or advanced techniques like Bayesian optimization for hyperparameter tuning.
* **Model Monitoring & Retraining**: Implement a system for monitoring model performance in production and a pipeline for periodic retraining with new data.
* **User Interface (UI)**: Develop a simple web interface for easier interaction and demonstration.
* **Database Integration**: Store item features, historical prices, and prediction logs for advanced analytics.
* **Authentication & Authorization**: Secure API endpoints.
* **Advanced CI/CD & Cloud Deployment**:
    * Fully automate build, test, and deployment using CI/CD tools (e.g., GitHub Actions).
    * Deploy to cloud platforms like AWS (ECS, EKS, App Runner), Google Cloud (Cloud Run, GKE), or Azure.
    * Set up infrastructure for logging, monitoring, and alerts in the cloud.
* **Batch Prediction Endpoint**: Implement an endpoint for efficient batch predictions.
* **A/B Testing Framework**: Develop capabilities to A/B test different pricing models or strategies.

---
## Contributing

Contributions are welcome! Please feel free to fork the repository, make changes, and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

-
## License

Distributed under the MIT License.






