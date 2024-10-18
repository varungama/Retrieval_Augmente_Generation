# Free Online Retrieval-Augmented Generation (RAG) backend system to analyse and compare two provided datasets of survey results

## Overview
This project integrates a server using Python and Langchain with FastAPI that acts as an intermediary between the frontend and the RAG system. It provides a framework for processing the datasets, interacting with the language model, and returning the results to your frontend using techniques like vector embeddings, semantic search, and prompt engineering to enhance the quality of the generated insights. 
It is specially designed to support Survey Excel files.

## Key Features
**Text Splitting:** Divides text into manageable chunks for embedding.

**Embeddings:** Generates text embeddings using the sentence-transformers/all-MiniLM-l6-v2 model from HuggingFace.
**Vector Store:** Stores and retrieves embeddings using FAISS DB.
**Interactive Q&A:** Allows users to ask questions and get responses based on the stored embeddings.

## Pre-Conditions
## Installation

**Install the necessary packages using:**
```bash
source bin/activate
pip install -r /path/to/requirements.txt
```

**Run the FastAPI application using Uvicorn:**
Go to the folder directory
```bash
cd RAG
uvicorn main:app --reload
```
You can access the app by navigating to http://127.0.0.1:8000 in your browser.



The model is ready to query on the Sustainability Research & Christmas Research Data.
If you want to use the custom survey data, please upload and preprocess it.
Since the model is running CPU it will take a while.

If you want to use GPU for fast processing.

## The commands to successfully install on windows (using cmd) are as follows:

```bash
set FORCE_CMAKE=1 && set CMAKE_ARGS=-DLLAMA_CUBLAS=on -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_FMA=off
pip install llama-cpp-python --no-cache-dir
```

You can remove -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_FMA=off (or set to on) if your hardware supports them.

To get the latest version from github if you don't want to rely on pip versions (usually lags a few days behind, sometimes more).
Run the following:
```bash
git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git llama-cpp-python-main
cd llama-cpp-python-main

set FORCE_CMAKE=1 && set "CMAKE_ARGS=-DLLAMA_CUBLAS=on -DLLAMA_AVX=off -DLLAMA_AVX2=off -DLLAMA_FMA=off"
python -m pip install .[all]
```
