# Sentiment Analysis Web Application

## About the Project

This project is a machine learning-based web application that analyzes text reviews and classifies their sentiment as Positive, Negative, or Neutral.

The application is built using Python and Streamlit. It allows users to analyze individual reviews and process multiple reviews through a CSV file.

## Features

* Sentiment classification into Positive, Negative, and Neutral.
* Single review analysis.
* Sentiment probability display.
* Bulk analysis using CSV files.
* Sentiment distribution charts.
* Downloadable results in CSV format.
* Interactive dashboard.

## Technologies Used

* Python
* Pandas
* Scikit-learn
* TF-IDF Vectorization
* Logistic Regression
* Streamlit
* Matplotlib
* Joblib

## Project Structure

```text
Sentiment_Analysis_Project/
├── app.py
├── sentiment_analysis.ipynb
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
└── README.md
```

## Machine Learning Workflow

1. Prepare and label the review dataset.
2. Split the data into training and testing sets.
3. Convert text into numerical features using TF-IDF.
4. Train a Logistic Regression classifier.
5. Evaluate the model on test data.
6. Save the trained model and vectorizer.
7. Use Streamlit to create the web application.

## Installation

Install the required libraries:

```bash
python -m pip install streamlit pandas scikit-learn matplotlib joblib
```

## How to Run

1. Open the project folder in VS Code.
2. Open the terminal in the project directory.
3. Run the following command:

```bash
python -m streamlit run app.py
```

4. Open the local address displayed in the terminal.

## Model Evaluation

The initial model achieved 50% accuracy on a small test set of 12 reviews.

This result is preliminary because the dataset is small. The model's performance may vary on new and unseen reviews.

## Limitations

* The model was trained on a small dataset.
* Predictions may be incorrect for unfamiliar phrases, sarcasm, or mixed-sentiment reviews.
* The model should not be considered reliable for real-world decisions without further training and evaluation.

## Future Improvements

* Train on a larger, diverse, labelled dataset.
* Improve model evaluation using precision, recall, and F1-score.
* Test performance on additional unseen reviews.
* Improve text preprocessing and model performance.

## Project Purpose

This project was developed for educational purposes to demonstrate text preprocessing, machine learning classification, model evaluation, and deployment using Streamlit.
