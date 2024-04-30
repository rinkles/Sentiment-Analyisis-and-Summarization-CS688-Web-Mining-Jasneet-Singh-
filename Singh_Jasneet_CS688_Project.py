import streamlit as st
import pdfplumber
from transformers import pipeline
import requests
from bs4 import BeautifulSoup
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Function for sentiment analysis using BERT
def analyze_sentiment(input_text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3) # Updated num_labels to 3 for positive, neutral, and negative
    inputs = tokenizer(input_text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)
    if prediction == 0:
        sentiment_label = 'negative'
    elif prediction == 1:
        sentiment_label = 'neutral'
    else:
        sentiment_label = 'positive'
    return sentiment_label

# Function to scrape text from a website
def fetch_text_from_website(url):
    response = requests.get(url)
    if response.status_code != 200:
        return "Error: Unable to fetch webpage", ""
    soup = BeautifulSoup(response.content, 'html.parser')
    content_text = ' '.join([p.text for p in soup.find_all('p')])
    summarized_content = summarize_text(content_text)
    return content_text, summarized_content

# Function to extract text from a PDF file
def extract_text_from_pdf_file(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        num_pages = len(pdf.pages)
        if num_pages == 0:
            return "Error: PDF file contains no pages", ""
        for page_number, page in enumerate(pdf.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e:
                return f"Error processing page {page_number + 1}: {e}", ""
    summarized_text = summarize_text(text)
    return text, summarized_text

# Function to summarize text
def summarize_text(input_text):
    summarizer = pipeline("summarization")
    max_input_length = 150
    if len(input_text) > max_input_length:
        input_text = input_text[:max_input_length]
    summary_result = summarizer(input_text, max_length=50, min_length=30, do_sample=False)
    return summary_result[0]['summary_text']

# Main function for Streamlit web app
def main():
    st.title("Sentiment Analysis and Text Summarization")
    option = st.selectbox("Choose Input Type:", ("URL", "PDF"))

    if option == "URL":
        input_url = st.text_input("Enter URL:")
        if st.button("Analyze"):
            text_content, summarized_text = fetch_text_from_website(input_url)
            if text_content:
                sentiment_result = analyze_sentiment(text_content)
                st.write("Sentiment Analysis:", sentiment_result)
                st.write("Summarized Text:", summarized_text)
            else:
                st.write("Error: Unable to fetch text from URL")

    elif option == "PDF":
        uploaded_pdf_file = st.file_uploader("Upload PDF", type=['pdf'])
        if uploaded_pdf_file is not None:
            if st.button("Analyze"):
                pdf_text, summarized_pdf_text = extract_text_from_pdf_file(uploaded_pdf_file)
                sentiment_result = analyze_sentiment(pdf_text)
                st.write("Sentiment Analysis:", sentiment_result)
                st.write("Summarized Text:", summarized_pdf_text)

if __name__ == "__main__":
    main()