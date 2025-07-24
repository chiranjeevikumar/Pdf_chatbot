import streamlit as st
def pdf_uploader():
    return st.file_uploader("upload a pdf file", type=["pdf"],accept_multiple_files=True,help="upload one or more medical documents")