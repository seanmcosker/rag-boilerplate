import os

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.output_parsers import StrOutputParser

from basic_chain import get_model
from rag_chain import make_rag_chain
from remote_loader import load_web_page
from splitter import split_documents
from vector_store import create_vector_db
from dotenv import load_dotenv
import streamlit as st
#from st_files_connection import FilesConnection

already_embedded = True


def ensemble_retriever_from_docs(docs, embeddings=None,already_embedded=True):
    texts = split_documents(docs)

    #if already_embedded:
     #   conn = st.connection("s3", type=FilesConnection)
        
        
    vs = create_vector_db(texts, embeddings)
    vs_retriever = vs.as_retriever()

    bm25_retriever = BM25Retriever.from_texts([t.page_content for t in texts])

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vs_retriever],
        weights=[0.5, 0.5])

    return ensemble_retriever