import streamlit as st
import pdfplumber
import os
import json
import re
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic import BaseModel
from typing import List

load_dotenv()

# Pydantic model for validation
class DocumentAnalysis(BaseModel):
    summary: str
    key_points: List[str]
    action_items: List[str]
    risk_flags: List[str]
    confidence_score: float

# Set up OpenRouter client via OpenAI SDK
api_key = os.getenv("OPENROUTER_API_KEY")
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
) if api_key else None

def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def analyze_chunk(chunk: str) -> dict:
    if not client:
        return {"error": "OpenRouter API key not configured."}
    
    system_prompt = """You are a document analyst. Respond ONLY in valid JSON with these exact fields: summary (string), key_points (list of strings), action_items (list of strings), risk_flags (list of strings), confidence_score (float 0-1)"""
    
    prompt = f"Analyze the following text chunk:\n\n{chunk}"
    
    try:
        # Prompting via OpenRouter
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini", # OpenRouter model name format
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
            # Validate with Pydantic
            validated_data = DocumentAnalysis(**data)
            return validated_data.model_dump()
        else:
            return {"error": "Could not parse JSON from OpenRouter response."}
    except Exception as e:
        return {"error": str(e)}

def main():
    st.set_page_config(page_title="Document Intelligence Tool", layout="wide")
    st.title("📄 Document Intelligence Tool")
    
    if not os.getenv("OPENROUTER_API_KEY"):
        st.warning("⚠️ Please set your OPENROUTER_API_KEY in the .env file to use the tool.")
    
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Analyze Document"):
            if not client:
                st.error("Cannot proceed without OpenRouter API key.")
                return

            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)
            
            if not text.strip():
                st.error("No text could be extracted from the uploaded PDF.")
                return
            
            with st.spinner("Splitting text into chunks..."):
                # Approximate 800 tokens by using ~3200 characters (roughly 4 chars/token)
                # Langchain's TokenTextSplitter can also be used but RecursiveCharacter is requested
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=3200, 
                    chunk_overlap=400,
                    length_function=len,
                )
                chunks = text_splitter.split_text(text)
            
            st.info(f"Document split into {len(chunks)} chunks for analysis.")
            
            all_results = []
            
            progress_bar = st.progress(0)
            for i, chunk in enumerate(chunks):
                with st.spinner(f"Analyzing chunk {i+1} of {len(chunks)}..."):
                    result = analyze_chunk(chunk)
                    if "error" in result:
                        st.error(f"Error in chunk {i+1}: {result['error']}")
                    else:
                        all_results.append(result)
                progress_bar.progress((i + 1) / len(chunks))
            
            st.success("Analysis complete!")
            
            # Display results
            st.header("Analysis Results")
            for i, result in enumerate(all_results):
                st.subheader(f"Chunk {i+1}")
                
                score = result['confidence_score']
                if score < 0.7:
                    st.error(f"⚠️ Low confidence score: {score}")
                else:
                    st.success(f"Confidence score: {score}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    with st.container(border=True):
                        st.markdown("### 📝 Summary")
                        st.write(result['summary'])
                    
                    with st.container(border=True):
                        st.markdown("### 🔑 Key Points")
                        for point in result['key_points']:
                            st.markdown(f"- {point}")
                
                with col2:
                    with st.container(border=True):
                        st.markdown("### ⚡ Action Items")
                        for item in result['action_items']:
                            st.markdown(f"- {item}")
                    
                    with st.container(border=True):
                        st.markdown("### 🚩 Risk Flags")
                        for flag in result['risk_flags']:
                            st.markdown(f"- {flag}")
                st.markdown("---")
            
            # Download as JSON
            if all_results:
                st.download_button(
                    label="Download Results as JSON",
                    file_name="analysis_results.json",
                    mime="application/json",
                    data=json.dumps(all_results, indent=2),
                )

if __name__ == "__main__":
    main()
