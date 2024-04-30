# Sentiment-Analyisis-and-Summarization-CS688-Web-Mining-Jasneet-Singh-
Web Mining Project
# Sentiment Analysis and Text Summarization App

This Streamlit web application allows users to perform sentiment analysis and text summarization on textual content obtained from either a URL or a PDF file.

## Features

- **Sentiment Analysis**: Utilizes the BERT (Bidirectional Encoder Representations from Transformers) model to classify the sentiment of the input text as positive, neutral, or negative.
- **Text Summarization**: Uses the Hugging Face `pipeline` for text summarization to generate a concise summary of the input text.

## Getting Started

1. **Clone the Repository**:

   ```bash
   git clone <"https://github.com/rinkles/Sentiment-Analyisis-and-Summarization-CS688-Web-Mining-Jasneet-Singh-/tree/main">
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:

   ```bash
   streamlit run Singh_Jasneet_CS688_Project.py
   ```

4. **Open the Browser**:

   Open your web browser and go to `http://localhost:8501` to access the application.

## Usage

- Choose the input type (URL or PDF) from the dropdown menu.
- Enter the URL or upload a PDF file.
- Click the "Analyze" button to trigger sentiment analysis and text summarization.
- View the sentiment analysis result and the summarized text.

## Technologies Used

- [Streamlit](https://streamlit.io/): For creating the interactive web application.
- [PDFplumber](https://github.com/jsvine/pdfplumber): For extracting text from PDF files.
- [Transformers](https://huggingface.co/transformers/): For sentiment analysis and text summarization using pre-trained models.
- [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/): For parsing HTML content when scraping from URLs.
