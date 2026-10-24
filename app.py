
import streamlit as st
import joblib
import pandas as pd

# Load saved model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="SentimentAI Dashboard",
    page_icon="💜",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.stApp {
    background-color: #0b0b12;
    color: #f5f3ff;
}

[data-testid="stSidebar"] {
    background-color: #151321;
    border-right: 1px solid #30234a;
}

h1, h2, h3 {
    color: #c4a1ff !important;
}

.subtitle {
    color: #aaa5bd;
    font-size: 16px;
}

.metric-card {
    background: linear-gradient(135deg, #191526, #24183b);
    border: 1px solid #49316c;
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 12px;
}

.metric-label {
    color: #b9b1ca;
    font-size: 14px;
}

.metric-value {
    color: #d6b8ff;
    font-size: 30px;
    font-weight: bold;
    margin-top: 8px;
}

div.stButton > button {
    background: linear-gradient(90deg, #7138c8, #a855f7);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: bold;
    width: 100%;
}

div.stButton > button:hover {
    border: 1px solid #d8b4fe;
    color: white;
}

div[data-baseweb="textarea"] {
    border: 1px solid #6941a5;
    border-radius: 10px;
}

hr {
    border-color: #30234a;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("# 💜 SentimentAI")
    st.caption("Review Intelligence Dashboard")
    st.divider()

    page = st.radio(
    "NAVIGATION",
    [
        "🏠 Dashboard",
        "📝 Analyze Review",
        "📂 Bulk Analysis",
        "ℹ️ About"
    ]
)
    st.divider()
    st.caption("Built with Python & Machine Learning")

# Dashboard page
if page == "🏠 Dashboard":

    st.title("💜 Sentiment Analysis Dashboard")
    st.markdown(
        '<p class="subtitle">Understand the sentiment behind your text.</p>',
        unsafe_allow_html=True
    )
    st.caption("✨ Powered by Machine Learning")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">💬 SENTIMENT CLASSES</div>
            <div class="metric-value">3 Classes</div>
            <div class="metric-label">Positive, Negative, Neutral</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">🤖 ML MODEL</div>
            <div class="metric-value">Logistic Regression</div>
            <div class="metric-label">Classification algorithm</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">📝 TEXT FEATURES</div>
            <div class="metric-value">{len(tfidf.vocabulary_)}</div>
            <div class="metric-label">TF-IDF vocabulary size</div>
        </div>
        """, unsafe_allow_html=True)


    st.info(
    
    "💡 Welcome to SentimentAI! Use Analyze Review to predict the "
    "sentiment of a single review, or Bulk Analysis to process multiple "
    "reviews from a CSV file."
)

    st.markdown("### 🚀 Quick Start")
    st.write("Choose **Analyze Review** from the sidebar to get started.")

# Analyze review page
elif page == "📝 Analyze Review":

    st.title("📝 Analyze a Review")
    st.markdown(
        '<p class="subtitle">Enter text and let the model predict its sentiment.</p>',
        unsafe_allow_html=True
    )

    user_text = st.text_area(
        "Your review",
        placeholder="Example: The product is amazing and I love it!",
        height=150
    )

    if st.button("✨ Analyze Sentiment"):

        if user_text.strip():

            text_tfidf = tfidf.transform([user_text])
            prediction = model.predict(text_tfidf)[0]

            st.divider()
            st.subheader("Analysis Result")

            if prediction == "Positive":
                st.success("😊 Positive Sentiment")

            elif prediction == "Negative":
                st.error("😞 Negative Sentiment")

            else:
                st.info("😐 Neutral Sentiment")

            st.markdown("### Prediction Details")
            st.write(f"**Predicted class:** {prediction}")
            st.write(f"**Characters analyzed:** {len(user_text)}")

            # Display class probabilities when supported
            if hasattr(model, "predict_proba") and hasattr(model, "classes_"):

                probabilities = model.predict_proba(text_tfidf)[0]

                st.markdown("### Sentiment Scores")

                probability_data = pd.DataFrame({
                    "Sentiment": model.classes_,
                    "Score (%)": probabilities * 100
                }).sort_values("Score (%)", ascending=False)

                for _, row in probability_data.iterrows():
                    st.write(
                        f"**{row['Sentiment']}** — {row['Score (%)']:.1f}%"
                    )
                    st.progress(float(row["Score (%)"] / 100))

                st.caption(
                    "These are model probability estimates, not guaranteed "
                    "measures of correctness. Model quality depends on "
                    "the training data."
                )

        else:
            st.warning("Please enter a review before analyzing.")

    
# Bulk analysis page
elif page == "📂 Bulk Analysis":

    st.title("📂 Bulk Sentiment Analysis")
    st.write("Upload a CSV file containing multiple reviews.")

    uploaded_file = st.file_uploader(
        "Upload your reviews CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        text_column = st.selectbox(
            "Select the column containing reviews",
            df.columns
        )

        if st.button("Analyze All Reviews"):

            reviews = df[text_column].fillna("").astype(str)
            valid_mask = reviews.str.strip().ne("")
            valid_reviews = reviews[valid_mask]

            if valid_reviews.empty:
                st.warning("The selected column contains no reviews.")

            else:
                review_tfidf = tfidf.transform(valid_reviews)
                predictions = model.predict(review_tfidf)

                result_df = df.loc[valid_mask].copy()
                result_df["Predicted Sentiment"] = predictions

                st.success(
                    f"Successfully analyzed {len(result_df)} reviews!"
                )

                st.subheader("Analysis Results")
                st.dataframe(result_df, use_container_width=True)

                counts = result_df["Predicted Sentiment"].value_counts()

                percentages = (
                    counts / len(result_df) * 100
                ).reindex(
                    ["Positive", "Negative", "Neutral"],
                    fill_value=0
                )

                st.subheader("📊 Sentiment Distribution")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("😊 Positive", f"{percentages['Positive']:.1f}%")

                with col2:
                    st.metric("😞 Negative", f"{percentages['Negative']:.1f}%")

                with col3:
                    st.metric("😐 Neutral", f"{percentages['Neutral']:.1f}%")

                st.bar_chart(percentages)

                csv_data = result_df.to_csv(index=False).encode("utf-8-sig")

                st.download_button(
                    "📥 Download Results CSV",
                    data=csv_data,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv"
                )


# About page
elif page == "ℹ️ About":

    st.title("ℹ️ About SentimentAI")

    st.write(
        "SentimentAI is a machine-learning application that classifies "
        "text into Positive, Negative, or Neutral sentiment."
    )

    st.markdown("### Technologies Used")

    st.markdown("""
    - **Python** — application logic
    - **Scikit-learn** — machine-learning model
    - **TF-IDF** — text feature extraction
    - **Streamlit** — web application
    - **Pandas** — displaying prediction scores
    - **HTML & CSS** — custom dashboard styling
    """)

    st.warning(
        "This is an educational project. Its current model was trained "
        "on a small sample dataset and may not generalize well to real reviews."
    )

