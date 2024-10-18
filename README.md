# Free Online Retrieval-Augmented Generation (RAG) backend system to analyse and compare two provided datasets of survey results

## Overview
This project integrates a server using Python and Langchain with FastAPI that acts as an intermediary between the frontend and the RAG system, providing a framework for  process the datasets, interact with the language model, and return the results to your frontend using techniques like vector embeddings, semantic search, and prompt engineering to enhance the quality of the generated insights. 
It is specially designed to support Survey Excel files.


('''set FORCE_CMAKE=1 && set CMAKE_ARGS=-DLLAMA_CUBLAS=on -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_FMA=off
pip install llama-cpp-python --no-cache-dir''')
