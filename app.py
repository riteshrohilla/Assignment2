import streamlit as st

openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

import openai
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
import tempfile
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    # Setting OpenAI API key here
OPENAI_API_KEY = 'sk-proj-iPONakH6eS0VCgyb43wlT3BlbkFJjCmSaksyqd5AbLJSb8yJ'
openai.api_key = OPENAI_API_KEY

st.title("RAG with ChatGPT")
st.write("Retrieval Augmented Generation (RAG) with GPT combines document retrieval and generative capabilities to provide contextually relevant answers by retrieving pertinent information from documents and using GPT to generate coherent responses.")

if 'messages' not in st.session_state:
    st.session_state.messages = []

def generate_response(prompt, retrieved_docs):
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": full_prompt},
        ],
        max_tokens=150,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content'].strip()

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)
    return sentiment

def retrieve_documents(query, retriever, k=2):
    return retriever.get_relevant_documents(query)[:k]

uploaded_file = st.file_uploader("Upload your document ", type=["pdf"])

if uploaded_file:
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(uploaded_file.read())
    temp_file.close()

    loader = PyMuPDFLoader(file_path=temp_file.name)
    docs = loader.load()
    os.remove(temp_file.name)

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)

    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()

    user_input = st.text_input("Please Enter Your Query: ", "")
    if st.button("Send"):
        st.session_state.messages.append(("User", user_input))
        retrieved_docs = retrieve_documents(user_input, retriever)
        response = generate_response(user_input, retrieved_docs)
        sentiment = analyze_sentiment(response)
        st.session_state.messages.append(("ChatGPT", response))
        st.session_state.messages.append(("Sentiment", sentiment))

for user, msg in st.session_state.messages:
    st.write(f"{user}: {msg}")
else:
    st.write("Please upload a document to proceed.")
