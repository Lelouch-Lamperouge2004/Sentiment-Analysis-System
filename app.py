import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import requests

import backend


API_URL = "https://sentiment-analysis-api-r6av.onrender.com"


st.set_page_config(
    page_title="Sentiment Analysis System",
    layout="wide"
)


@st.cache_data
def load_data():
    return backend.load_dataset()


@st.cache_data
def load_metrics():
    return backend.get_production_metrics()


def api_predict(review):
    response = requests.post(
        f"{API_URL}/predict",
        json={"review": review},
        timeout=10
    )
    response.raise_for_status()
    return response.json()


def api_batch_predict(reviews):
    response = requests.post(
        f"{API_URL}/predict-batch",
        json={"reviews": reviews},
        timeout=60
    )
    response.raise_for_status()
    return response.json()


def api_history():
    response = requests.get(f"{API_URL}/history", timeout=10)
    response.raise_for_status()
    return response.json()


def api_analytics():
    response = requests.get(f"{API_URL}/analytics", timeout=10)
    response.raise_for_status()
    return response.json()


df = load_data()

st.sidebar.title("Sentiment Analysis")
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Predict Review",
        "Batch Prediction",
        "Model Evaluation",
        "Prediction Analytics"
    ]
)

st.title("Sentiment Analysis System")
st.write(
    "A full-stack NLP application using Streamlit, FastAPI, Neon PostgreSQL, "
    "SQLAlchemy, Scikit-learn, and a saved Linear SVC model."
)


if page == "Dashboard":
    st.subheader("Dashboard")

    total_reviews = len(df)
    positive_reviews = int((df["feedback"] == 1).sum())
    negative_reviews = int((df["feedback"] == 0).sum())
    metrics, report, cm = load_metrics()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Training Reviews", total_reviews)
    col2.metric("Positive Reviews", positive_reviews)
    col3.metric("Negative Reviews", negative_reviews)
    col4.metric("Model", "Linear SVC")

    col5, col6, col7 = st.columns(3)
    col5.metric("Accuracy", round(metrics["Accuracy"], 4))
    col6.metric("Negative F1", round(metrics["Negative F1"], 4))
    col7.metric("Positive F1", round(metrics["Positive F1"], 4))

    fig = go.Figure(
        data=[
            go.Bar(
                x=["Negative", "Positive"],
                y=[negative_reviews, positive_reviews]
            )
        ]
    )

    fig.update_layout(
        title="Balanced Training Dataset",
        xaxis_title="Sentiment",
        yaxis_title="Review Count"
    )

    st.plotly_chart(fig, use_container_width=True)


elif page == "Predict Review":
    st.subheader("Single Review Prediction")

    user_input = st.text_area(
        "Enter a review",
        placeholder="Example: The product quality is excellent."
    )

    if st.button("Predict Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter a review before prediction.")
        else:
            try:
                result = api_predict(user_input)

                st.success("Prediction completed and saved to database.")

                col1, col2 = st.columns(2)
                col1.metric("Predicted Sentiment", result["sentiment"])
                col2.metric("Confidence Score", f"{result['confidence']}%")

                st.write("Input Review")
                st.info(user_input)

                st.write("Processed Text")
                st.code(result["processed_text"])

            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")


elif page == "Batch Prediction":
    st.subheader("Batch CSV Prediction")

    st.write("Upload a CSV file with a column named `review`.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            upload_df = pd.read_csv(uploaded_file)

            if "review" not in upload_df.columns:
                st.error("CSV must contain a column named `review`.")
            else:
                st.success("CSV uploaded successfully.")
                st.dataframe(upload_df.head(), use_container_width=True)

                if st.button("Run Batch Prediction"):
                    reviews = upload_df["review"].dropna().astype(str).tolist()
                    batch_results = api_batch_predict(reviews)
                    result_df = pd.DataFrame(batch_results)

                    st.subheader("Prediction Results")
                    st.dataframe(result_df, use_container_width=True)

                    csv_data = result_df.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Results CSV",
                        data=csv_data,
                        file_name="sentiment_predictions.csv",
                        mime="text/csv"
                    )

        except Exception as e:
            st.error(f"Error: {e}")


elif page == "Model Evaluation":
    st.subheader("Model Evaluation")

    if st.button("Compare Models"):
        comparison_df = backend.compare_models()
        st.dataframe(comparison_df, use_container_width=True)

    st.divider()

    if st.button("Evaluate Production Model"):
        metrics, report, cm = backend.get_production_metrics()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", round(metrics["Accuracy"], 4))
        col2.metric("Precision", round(metrics["Precision"], 4))
        col3.metric("Recall", round(metrics["Recall"], 4))
        col4.metric("F1 Score", round(metrics["F1 Score"], 4))

        cm_df = pd.DataFrame(
            cm,
            index=["Actual Negative", "Actual Positive"],
            columns=["Predicted Negative", "Predicted Positive"]
        )

        st.subheader("Confusion Matrix")
        st.dataframe(cm_df, use_container_width=True)

        fig, ax = plt.subplots()
        ax.imshow(cm)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Negative", "Predicted Positive"])
        ax.set_yticklabels(["Actual Negative", "Actual Positive"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center")

        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)


elif page == "Prediction Analytics":
    st.subheader("Prediction Analytics")

    try:
        analytics = api_analytics()
        history = api_history()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Predictions", analytics["total_predictions"])
        col2.metric("Positive Predictions", analytics["positive_predictions"])
        col3.metric("Negative Predictions", analytics["negative_predictions"])
        col4.metric("Average Confidence", f"{analytics['average_confidence']}%")

        if history:
            history_df = pd.DataFrame(history)

            st.subheader("Prediction Sentiment Distribution")

            sentiment_counts = history_df["sentiment"].value_counts()

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=sentiment_counts.index,
                        y=sentiment_counts.values
                    )
                ]
            )

            fig.update_layout(
                xaxis_title="Sentiment",
                yaxis_title="Prediction Count"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Recent Predictions")
            st.dataframe(
                history_df[
                    ["review", "sentiment", "confidence", "processed_text", "created_at"]
                ],
                use_container_width=True
            )
        else:
            st.info("No predictions stored yet.")

    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")