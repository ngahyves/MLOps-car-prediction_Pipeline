# End-to-End MLOps Pipeline for Used Car Price Estimation

![image](https://github.com/user-attachments/assets/59650abe-163d-4f7f-9fcd-f8aa97562b03)
 
The data set is available on kaggle: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data

## 🌟 Overview

This project implements a complete, production-grade MLOps pipeline to predict the price of used cars. It demonstrates a full-cycle machine learning system, from raw data ingestion to a deployed, interactive web application. The entire workflow is designed to be automated, reproducible, and scalable, simulating a real-world enterprise environment.

The core idea is not just to build a model, but to build a **robust system** that can reliably serve that model. This project covers key skills in Data Engineering, Machine Learning, and DevOps (MLOps).

### Key Features:
- **Real-World Problem:** Solves the universal business problem of asset valuation.
- **Full MLOps Lifecycle:** Covers data ingestion, preprocessing, experiment tracking, model containerization, CI/CD, and deployment.
- **Production-Ready Simulation:** The pipeline mimics a professional workflow: Raw Data -> Database -> Preprocessing -> Training -> Deployment.
- **Diverse Skillset:** Showcases expertise in database management (SQLite), regression modeling, software engineering (REST API), and MLOps practices (MLflow, Docker, GitHub Actions).

---

## 🛠️ Tech Stack

- **Data Processing & Modeling:** `Python`, `Pandas`, `Scikit-learn`, `XGBoost`
- **Database:** `SQLite`
- **ML Experiment Tracking:** `MLflow`
- **API Development:** `FastAPI`
- **Containerization:** `Docker`, `Docker Compose`
- **CI/CD Automation:** `GitHub Actions`
- **Interactive Dashboard:** `Streamlit`

---

## 🚀 Project Architecture

![Screenshot 2025-07-02 164109](https://github.com/user-attachments/assets/ede7d169-52bc-4181-aeb6-6af1c43aab88)


The project is structured as an automated pipeline, where each component is a modular script or a containerized service.


 <!-- Un diagramme simple serait un énorme plus ! Vous pouvez utiliser des outils comme diagrams.net (draw.io) -->

### The Pipeline Stages:

1.  **Data Ingestion (`ingest_data.py`):** Raw data from a Kaggle CSV file is loaded into a centralized SQLite database (`cars.db`). This step is idempotent, ensuring a clean and consistent starting point for the pipeline.
2.  **Preprocessing & Feature Engineering (`preprocess.py`):** Data is cleaned, and new features (e.g., `car_age`) are engineered. Categorical variables are transformed using one-hot encoding, and the processed data is prepared for model training.
3.  **Model Training & Experiment Tracking (`train.py`):**
    -   Multiple regression models (e.g., Linear Regression, RandomForest, XGBoost) are trained.
    -   **MLflow** is used to log every experiment, tracking parameters, metrics (R², MAE, RMSE), and artifacts (the model itself).
    -   The best-performing model is identified and saved as `best_model.pkl`.
4.  **API Deployment (`api.py` & `Dockerfile`):**
    -   A **FastAPI** REST API is created to serve the `best_model.pkl`. It exposes a `/predict` endpoint that takes car features as input and returns a price estimation.
    -   The API is containerized using **Docker**, creating a portable and isolated environment for deployment.
5.  **Continuous Integration & Delivery (CI/CD):**
    -   A **GitHub Actions** workflow automates testing and deployment.
    -   **CI:** On every push, the workflow installs dependencies, runs tests, builds the Docker image, and pushes it to a container registry (e.g., Docker Hub).
    -   **CD (Optional):** Automatically deploys the new image to a cloud platform.
6.  **User Interface & Orchestration (`dashboard.py` & `docker-compose.yml`):**
    -   A user-friendly **Streamlit** dashboard provides an interface for real-time price predictions by calling the FastAPI backend.
    -   **Docker Compose** orchestrates all services (API, Dashboard), allowing the entire multi-container application to be launched with a single command.

---

## 🏁 How to Run This Project Locally

### Prerequisites
- Docker and Docker Compose installed.
- Git.
- A Kaggle account to download the dataset (or provide the dataset file).

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

### Step 2: Set up the Data
-Download the "Used Car Dataset" from Kaggle and place the CSV file in a data/ directory at the root of the project.
-You might need to update the path in ingest_data.py if necessary.

### Step 3: Run the Full Pipeline Manually (Step-by-Step)
This allows you to see each component in action.
#### a-Ingest Data into the Database:

bash
python src/ingest_data.py

#### b-Preprocess the Data:
bash
python src/preprocess.py

#### c-Train the Model and Track with MLflow:
bash
python src/train.py


#### d-To view the experiments, run the MLflow UI in a separate terminal:
bash
mlflow ui

bash
Then, open http://127.0.0.1:5000 in your browser.

### Step 4: Run the Full Application with Docker Compose
This is the easiest way to see the final product. It will build the images for the API and the dashboard and run them as interconnected services.
bash
docker-compose up --build

Once the services are running:
API: Access the interactive API documentation at http://localhost:8000/docs.
Dashboard: Interact with the user-friendly application at http://localhost:8501.

📂 Project Structure
Generated code
.
├── .github/workflows/      # CI/CD pipeline configuration
├── data/                   # Raw data (e.g., from Kaggle)
├── src/                    # Source code for the pipeline
│   ├── ingest_data.py
│   ├── preprocess.py
│   ├── train.py
│   ├── api.py
│   └── dashboard.py
├── best_model.pkl          # Saved model from the training script
├── cars.db                 # SQLite database
├── docker-compose.yml      # Orchestration for all services
├── Dockerfile              # Docker configuration for the API
├── mlruns/                 # MLflow experiment tracking data
├── requirements.txt        # Python dependencies
└── README.md

✨ Future Improvements
Deploy the CD pipeline to a cloud provider like AWS or Heroku.
Implement data and model monitoring to detect drift.
Add more comprehensive unit and integration tests.
Switch the backend store for MLflow to a more robust database like PostgreSQL.
Generated code

