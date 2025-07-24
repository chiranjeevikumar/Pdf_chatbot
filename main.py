import streamlit as st
from app.ui import pdf_uploader
from app.pdf_utils import extract_text_from_pdf
from app.vectorstores_utils import create_faiss_index, retrieve_relevant_docs
from app.chat_utils import get_chat_model, ask_chat_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time

st.set_page_config(page_title="Medical Chat assistant",page_icon=":hospital:",layout="wide",initial_sidebar_state="expanded")
# Custom CSS styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stTextInput input {
        border-radius: 5px;
        border: 1px solid #ddd;
        padding: 0.5rem;
    }
    .css-1d391kg {  /* Chat containers */
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f5f5f5;
    }
    .stMarkdown {
        font-size: 16px;
        line-height: 1.6;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_model" not in st.session_state:
    st.session_state.chat_model = None

with st.sidebar:
    st.markdown("#### Document upload")
    st.markdown("Upload one or more medical documents to start chatting.")
    uploaded_files = pdf_uploader()
    if uploaded_files:
        st.success(f"{len(uploaded_files)} document(s) uploaded successfully.")
        if st.button("Process Documents", type="primary"):
            with st.spinner("Processing your medical documents..."):
                all_texts=[]
                for file in uploaded_files:
                    text = extract_text_from_pdf(file)
                    all_texts.append(text)
                text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = []
                for text in all_texts:
                    chunks.extend(text_splitter.split_text(text))
                vectorstore = create_faiss_index(chunks)
                st.session_state.vectorstore = vectorstore
                st.success("Document processing complete.")
                chat_model = get_chat_model()
                st.session_state.chat_model = chat_model
                st.success("Documents processed successfully")
                st.balloons()

st.markdown("#### Chat with Your Medical Documents")

for messages in st.session_state.messages:
    with st.chat_message(messages["role"]):
        st.markdown(messages["content"])
        st.caption(messages["timestamp"])

if prompt := st.chat_input("Ask a question about your medical documents:"):

    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(time.strftime("%Y-%m-%d %H:%M:%S"))
    if st.session_state.vectorstore and st.session_state.chat_model:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                relevant_docs = retrieve_relevant_docs(st.session_state.vectorstore, prompt)
                context = "\n".join([doc.page_content for doc in relevant_docs])
                system_prompt = f"""
                You are a helpful medical assistant.
                Use the following context to answer the question:
                medical  Documents :
                {context}
                USer Question : {prompt}
                Answer : """
            
                response = ask_chat_model(st.session_state.chat_model, system_prompt)
            st.markdown(response)
            st.caption(time.strftime("%Y-%m-%d %H:%M:%S"))
            st.session_state.messages.append({"role": "assistant", "content": response, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
else:
    with st.chat_message("assistant"):
        st.error("Please upload documents and process them first.")
        st.caption(time.strftime("%Y-%m-%d %H:%M:%S"))


