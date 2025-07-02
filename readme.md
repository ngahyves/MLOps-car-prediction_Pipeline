# End-to-End MLOps Pipeline for Used Car Price Estimation

![Project GIF](https://github.com/user-attachments/assets/59650abe-163d-4f7f-9fcd-f8aa97562b03)

## 🌟 Overview

This project implements a complete, production-grade MLOps pipeline to predict the price of used cars. It demonstrates a full-cycle machine learning system, from raw data ingestion to a deployed, interactive web application. The entire workflow is designed to be automated, reproducible, and scalable, simulating a real-world enterprise environment.

The core idea is not just to build a model, but to build a **robust system** that can reliably serve that model. This project covers key skills in Data Engineering, Machine Learning, and DevOps (MLOps).

### Key Features:
- **Real-World Problem:** Solves the universal business problem of asset valuation.
- **Full MLOps Lifecycle:** Covers data ingestion, preprocessing, experiment tracking, model containerization, CI/CD, and deployment orchestration.
- **Production-Ready Simulation:** The pipeline mimics a professional workflow: Raw Data -> Database -> Preprocessing -> Training -> Deployment.
- **Diverse Skillset:** Showcases expertise in database management (SQLite), regression modeling, software engineering (REST API), and MLOps practices (MLflow, Docker, GitHub Actions).

---

## 🛠️ Tech Stack

- **Data Processing & Modeling:** `Python`, `Pandas`, `Scikit-learn`, `XGBoost`
- **Database:** `SQLite`
- **ML Experiment Tracking:** `MLflow`
- **API Development:** `FastAPI`
- **Containerization & Orchestration:** `Docker`, `Docker Compose`
- **CI/CD Automation:** `GitHub Actions`
- **Interactive Dashboard:** `Streamlit`

---

## 🚀 Project Architecture

The project is structured as an automated pipeline where each component is a modular script or a containerized service.

![Architecture Diagram](https://github.com/user-attachments/assets/ede7d169-52bc-4181-aeb6-6af1c43aab88)

### The Pipeline Stages:
1.  **Data Ingestion (`ingest_data.py`):** Raw data from a Kaggle CSV file is loaded into a centralized SQLite database (`cars.db`).
2.  **Preprocessing (`preprocess.py`):** Data is cleaned, and features are engineered for model training.
3.  **Model Training & Tracking (`train.py`):** Multiple models are trained and their performance is tracked with **MLflow**. The best model is saved.
4.  **API Serving (`api.py` & `Dockerfile`):** A **FastAPI** REST API serves the best model. The API is containerized with **Docker**.
5.  **CI/CD Automation (`.github/workflows/ci.yml`):** A **GitHub Actions** workflow automatically builds the Docker image and pushes it to Docker Hub on every change to the `main` branch.
6.  **User Interface & Orchestration (`dashboard.py` & `docker-compose.yml`):** A **Streamlit** dashboard provides a user-friendly interface for predictions. **Docker Compose** orchestrates both the API and dashboard services, allowing them to run together seamlessly.

---

## 🏁 Getting Started

### Prerequisites
- Docker & Docker Compose
- Git
- A web browser

### Run the Full Application
This is the recommended way to run the project. The single command below will build the necessary Docker images and launch the interconnected API and dashboard services.

1.  **Clone the repository:**
    ```shell
    git clone https://github.com/ngahyves/MLOps-car-prediction_Pipeline.git
    cd MLOps-car-prediction_Pipeline
    ```

2.  **Generate the model file:**
    Since the trained model is not versioned in Git, you need to generate it locally first. Run the training script in "fast mode" to create the `best_model.pkl` file in a few minutes.
    ```shell
    python train.py --mode fast
    ```

3.  **Run the application with Docker Compose:**
    ```shell
    docker-compose up --build
    ```

4.  **Interact with the services:**
    -   **Interactive Dashboard:** Open your browser and go to `http://localhost:8501`
    -   **API Documentation:** Access the FastAPI docs at `http://localhost:8000/docs`

---

## 📂 Project Structure
.
├── .github/workflows/ # CI/CD pipeline for Docker image automation
├── models/ # Output for the trained model (ignored by Git)
├── .dockerignore # Specifies files to ignore for Docker builds
├── .gitignore # Specifies files to ignore for Git
├── api.py # FastAPI application script
├── dashboard.py # Streamlit dashboard script
├── docker-compose.yml # Orchestration for all services
├── Dockerfile # Docker configuration for the API
├── Dockerfile.dashboard # Docker configuration for the Dashboard
├── ingest_data.py # Script for data ingestion
├── preprocess.py # Script for data preprocessing
├── train.py # Script for model training (configurable with --mode)
├── requirements.txt # Python dependencies for the API
├── requirements.dashboard.txt # Python dependencies for the Dashboard
└── README.md
> **Note on Data and Models:** The raw data (`vehicles.csv`), the database (`cars.db`), the MLflow tracking data (`mlruns/`), and the final model (`models/best_model.pkl`) are not versioned in this repository as per best practices. They are generated locally when you run the pipeline scripts.

---

## ✨ Future Improvements
- Deploy the entire application to a cloud provider like AWS or Heroku using a CD (Continuous Deployment) workflow.
- Implement data and model monitoring to detect drift over time.
- Add more comprehensive unit and integration tests for the API and data processing steps.
- Switch the MLflow backend store to a more robust database like PostgreSQL for multi-user collaboration.
