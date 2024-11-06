import streamlit as st
from transformers import pipeline
import json
import os
import matplotlib.pyplot as plt 
 

#File to store reviews
REVIEW_FILE = "reviews.json"

#Function to load existing reviews
def load_reviews():
    if not os.path.exists(REVIEW_FILE):
        return []
    with open(REVIEW_FILE, "r") as file:
        return json.load(file)

def save_review(product, review, label, score):
    reviews = load_reviews()
    new_review = {
        "product" : product,
        "review" : review,
        "label" : label,
        "score" : score

    }
    reviews.append(new_review)
    with open(REVIEW_FILE, "w") as file: 
        json.dump(reviews, file, indent=4)

# Load the pre-trained sentiment analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

# Title of the app
st.title("Sentiment Analysis for Product Reviews")

products = ["Product A", "Product B", "Product C"]

col1, col2= st.columns(2)

with col1:
    selected_product = st.selectbox("Select a product to review:", products)
    # Input field for the user to enter a product review
    review = st.text_area(f"Enter your product review for {selected_product}:")




    # Analyze button
    if st.button("Analyze Sentiment"):
        if review:
            # Perform sentiment analysis using BERT
            result = sentiment_pipeline(review)[0]
            label = result['label']
            score = result['score']
        
            # Display the result
            if label == "POSITIVE":
                st.success(f"Sentiment: {label} with confidence {score:.2f}")
            else:
                st.error(f"Sentiment: {label} with confidence {score:.2f}")
        
            save_review(selected_product, review, label, score)
        else:
            st.write("Please enter a review to analyze.")


    filter_option = st.radio("View reviews:", ("All", "Positive", "Negative"))


    st.subheader(f"Reviews for {selected_product}:")
    st.write("---")
    all_reviews = load_reviews()
    product_reviews = [r for r in all_reviews if r['product'] == selected_product]

    if filter_option == "Positive":
        product_reviews = [r for r in product_reviews if r['label'] == "POSITIVE"]
    elif filter_option == "Negative": 
        product_reviews = [r for r in product_reviews if r['label'] == "NEGATIVE"]

    if product_reviews:
        for r in product_reviews:
            st.write(f"**Review:** {r['review']}")
            st.write(f"**Sentiment:** {r['label']} (Confidence: {r['score']:.2f})")
            st.write("---")
    else:
        st.write("No reviews yet for this product.")

with col2:
    show_statistics = st.checkbox("Show Review Statistics")
    if filter_option == "All" and show_statistics:
        positive_reviews = sum(1 for r in product_reviews if r['product'] == selected_product and r['label'] == "POSITIVE")
        negative_reviews = sum(1 for r in product_reviews if r['product'] == selected_product and r['label'] == "NEGATIVE")

         # Apply a Matplotlib style
        plt.style.use("ggplot")  # You can try "fivethirtyeight", "seaborn-darkgrid", etc.

        # Create a bar chart
        fig, ax = plt.subplots()
        ax.bar(["Positive", "Negative"], [positive_reviews, negative_reviews], color=["#2ca02c", "#d62728"])
        ax.set_ylabel("Number of Reviews")
        ax.set_title(f"Sentiment Distribution for {selected_product}")
        ax.grid(True, linestyle='--', alpha=0.7)  # Add grid lines for readability
        ax.set_facecolor("#f7f7f7")  # Set background color


        st.pyplot(fig)


