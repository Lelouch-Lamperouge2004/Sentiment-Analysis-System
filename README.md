# Sentiment Analysis Web Application using Streamlit

## Overview

The Sentiment Analysis Web Application is a machine learning–based system designed to analyze and classify the sentiment of textual reviews. The application helps businesses, researchers, and individuals understand user opinions by categorizing text data into Positive or Negative sentiment using Natural Language Processing (NLP) techniques.

The project is built using Python and Streamlit, providing an interactive web-based dashboard that allows users to visualize sentiment distributions and perform real-time sentiment predictions on custom text input.

## Live Application

The application is deployed on Streamlit Community Cloud and can be accessed using the link below:

https://sentiment-analysis-app-wlhpdgtxsrujcuepwjimzk.streamlit.app

## Features

- Interactive web-based dashboard built with Streamlit
- Text preprocessing using NLP techniques (tokenization, stopword removal, stemming)
- Sentiment visualization using Word Cloud and sentiment distribution bar chart
- Real-time sentiment prediction for user-entered text
- Multiple machine learning models for sentiment classification
  - Bernoulli Naive Bayes
  - Logistic Regression
  - Gradient Boosting Classifier
  - Linear Support Vector Classifier (Linear SVC)
- Clean and minimal user interface
- Deployed for real-time access using Streamlit Cloud

## Technologies Used

- Programming Language: Python
- Web Framework: Streamlit
- Data Processing: Pandas, NumPy
- Natural Language Processing: NLTK
- Machine Learning: Scikit-learn
- Visualization: Matplotlib, Plotly, WordCloud

## Project Structure

sentiment-analysis-app/
<br>├── app.py
<br>├── backend.py
<br>├── requirements.txt
<br>├── README.md
<br>└── .gitignore

## How the System Works

1. Text reviews are preprocessed using NLP techniques such as tokenization, stopword removal, and stemming.
2. The cleaned text is converted into numerical features using Count Vectorization.
3. Machine learning models are trained on labeled review data.
4. The trained models predict sentiment for new user-provided text.
5. Results are displayed through an interactive Streamlit dashboard.

## How to Run the Project Locally

1. Clone the repository  

2. Create a virtual environment (optional but recommended)  
   python -m venv venv  
   venv\Scripts\activate  

3. Install dependencies  
   pip install -r requirements.txt  

4. Run the application  
   streamlit run app.py  

The application will open automatically in your web browser.

## Use Cases

- Customer feedback analysis
- Product review analysis
- Opinion mining
- Academic and learning projects
- Business sentiment insights

## Future Enhancements

- Add neutral sentiment classification
- Integrate deep learning models such as LSTM and BERT
- Allow CSV file upload for bulk analysis
- Add sentiment trend analysis over time

## Conclusion

The Sentiment Analysis Web Application using Streamlit provides an easy-to-use and effective solution for understanding textual sentiment. With its clean interface, real-time predictions, and visual insights, the project demonstrates the practical application of machine learning and NLP concepts in a real-world scenario.
