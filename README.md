# Document Intelligence Tool

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenRouter](https://img.shields.io/badge/OpenRouter-API-blue?style=for-the-badge)

A powerful Python and Streamlit-based application designed to extract and analyze text from PDF documents using Large Language Models via OpenRouter (e.g., GPT-4o-mini, Claude, etc.).

## Features

- **PDF Upload & Extraction**: Seamlessly upload PDF files and accurately extract text using `pdfplumber`.
- **Intelligent Chunking**: Splits large documents into manageable chunks (approx. 800 tokens) using LangChain's `RecursiveCharacterTextSplitter`.
- **Advanced Document Analysis**: Leverages LLMs via OpenRouter to analyze each text chunk, outputting structured JSON including:
  - Summary
  - Key Points
  - Action Items
  - Risk Flags
  - Confidence Score
- **Strict Data Validation**: Uses `Pydantic` to ensure the AI's response strictly adheres to the requested JSON schema.
- **Interactive UI**: Clean Streamlit cards display the results. Features a dynamic warning banner for confidence scores below 0.7.
- **Export Options**: Download the fully structured analysis as a JSON file.

## Prerequisites

- Python 3.8+
- An [OpenRouter API Key](https://openrouter.ai/)

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/userjaggu/LLM_Based-Document-intelligence-tool.git
   cd LLM_Based-Document-intelligence-tool
   ```

2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install required dependencies:**
   ```bash
   pip install streamlit pdfplumber python-dotenv langchain-text-splitters pydantic openai
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory and add your OpenRouter API Key:
   ```env
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

## Running the Application

Start the Streamlit server with:

```bash
streamlit run app.py
```

The application will automatically open in your default web browser (typically at `http://localhost:8501`).

## How to Use

1. Open the app in your browser.
2. Ensure your `.env` file is set up correctly (a warning banner will show if the key is missing).
3. Upload a target PDF document using the file uploader.
4. Click **"Analyze Document"**.
5. Wait for the extraction, token chunking, and AI analysis to finish.
6. Review the generated Summary, Key Points, Action Items, and Risk Flags.
7. Note any red warning banners for chunks with a confidence score `< 0.7`.
8. Click **"Download Results as JSON"** to save your structured analysis.
