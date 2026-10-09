# Fake News Detection Using Machine Learning and Flask API

A machine learning project that classifies news-related data as **Real News** or **Fake News** using a Random Forest Classifier. The trained model is deployed through a Flask REST API and containerized using Docker.

## Project Overview

Fake news can spread misinformation through online platforms and social media. This project demonstrates how machine learning can classify news-related records using numerical features such as source credibility, author credibility, clickbait score, article length, and social shares.

The model is integrated with a Flask API, allowing users to submit feature values through an HTTP POST request and receive a prediction in JSON format.

> **Note:** This project uses structured numerical features, not the full text of news articles. Its predictions are model outputs and should not be treated as independent verification of factual accuracy.

## Technologies Used

- **Python** – Programming language
- **Pandas** – Data handling and preprocessing
- **Scikit-learn** – Machine learning and model evaluation
- **Random Forest Classifier** – Classification algorithm
- **Joblib** – Saving and loading the trained model
- **Flask** – REST API development
- **Gunicorn** – WSGI server
- **Postman** – API testing
- **Docker** – Containerization

## Dataset

The project uses the dataset:

`fake_news_detection - fake_news_detection.csv`

The dataset contains 1,000 records and 12 columns.

### Features Used

| Feature | Description |
|---|---|
| `source_credibility` | Credibility score of the source |
| `author_credibility` | Credibility score of the author |
| `clickbait_score` | Measure of clickbait characteristics |
| `sentiment_score` | Sentiment-related score |
| `article_length` | Length of the article |
| `num_quotes` | Number of quotes |
| `num_images` | Number of images |
| `num_links` | Number of links |
| `social_shares` | Number of social shares |
| `verified_source` | Source verification indicator |

**Target variable:** `label`

- `0` – Real News
- `1` – Fake News

The `news_type` column is excluded from model training.

## Machine Learning Workflow

1. Load the dataset using Pandas.
2. Separate the features (`X`) and target (`y`).
3. Split the data into training and testing sets using an 80:20 ratio.
4. Handle missing numerical values using median imputation.
5. Train a Random Forest Classifier.
6. Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
7. Save the trained pipeline as `fake_news_model.pkl`.

## Project Structure

```text
Fake_News_Detection_Prediction/
│
├── app.py
├── train_model.py
├── requirements.txt
├── Dockerfile
├── fake_news_model.pkl
├── fake_news_detection - fake_news_detection.csv
└── README.md
```

The `venv/` directory is used for local development and should not be uploaded to GitHub.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Fake_News_Detection_Prediction
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

Ensure the CSV dataset is present in the project directory, then run:

```bash
python3 train_model.py
```

This creates the file `fake_news_model.pkl`.

### 5. Run the Flask API locally

```bash
python3 app.py
```

The API runs on:

`http://127.0.0.1:5001`

## API Endpoints

### Home Endpoint

**GET** `/`

Checks whether the API is running.

Example response:

```json
{
  "endpoint": "/predict",
  "message": "Fake News Detection API is running!",
  "method": "POST"
}
```

### Prediction Endpoint

**POST** `/predict`

Accepts numerical news-related features and returns a classification prediction.

**Request URL:**

```text
http://127.0.0.1:5001/predict
```

**Request body:**

```json
{
  "source_credibility": 0.8,
  "author_credibility": 0.9,
  "clickbait_score": 0.2,
  "sentiment_score": 0.1,
  "article_length": 850,
  "num_quotes": 4,
  "num_images": 2,
  "num_links": 5,
  "social_shares": 120,
  "verified_source": 1
}
```

In Postman, select:

- Method: `POST`
- Body: `raw`
- Format: `JSON`

**Example response:**

```json
{
  "prediction": 0,
  "result": "Real News"
}
```

The actual result depends on the trained model and supplied feature values.

## Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The previously observed test accuracy was approximately **79%**. Re-run `train_model.py` to verify the metrics for the current version of your code and dataset.

## Docker Deployment

Docker packages the Flask application, its dependencies, and the trained model into a container.

### 1. Build the Docker image

Ensure Docker Desktop is running, then execute:

```bash
docker build -t fake-news-api .
```

### 2. Run the container

```bash
docker run -d --name fake-news-container -p 5001:5001 fake-news-api
```

### 3. Check the running container

```bash
docker ps
```

### 4. View container logs

```bash
docker logs fake-news-container
```

### 5. Stop the container

```bash
docker stop fake-news-container
```

To remove the stopped container:

```bash
docker rm fake-news-container
```

After starting the container, test the home and prediction endpoints in your browser and Postman using `http://127.0.0.1:5001`.

## Key Learning Outcomes

- Data preprocessing and feature selection
- Training and evaluating a classification model
- Building a machine learning pipeline
- Saving and loading models using Joblib
- Developing REST APIs with Flask
- Testing POST requests using Postman
- Containerizing an ML application with Docker
- Running a model-serving API inside a container

## Limitations and Future Improvements

- Use a larger, well-validated dataset for stronger generalization.
- Add text-based features using TF-IDF or transformer-based NLP models.
- Improve validation and hyperparameter tuning.
- Add automated unit and API tests.
- Add request validation and production-grade error handling.
- Deploy the container to a cloud platform.

## Author

**Laxita Gaur**

Machine Learning | Python | Flask | Docker

---

If you find this project useful, feel free to ⭐ the repository.
