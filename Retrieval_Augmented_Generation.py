from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
# from llama_cpp import Llama
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from custom_logger import get_logger


class Rag_model:
    def __init__(self, log_filename, folder_path=r'output', device = "cpu", search_type="similarity_score_threshold"):
        self.logger = get_logger(log_filename)
        self.path = folder_path
        self.search_type = search_type
        self.device = device
        self.rag_modelpath = self.download_model()

    def download_model(self):
        try:
            from huggingface_hub import hf_hub_download
            # Define the model repository and file
            model_repo = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
            file_name = "mistral-7b-instruct-v0.1.Q5_K_M.gguf"
            # Download the model file
            model_path = hf_hub_download(repo_id=model_repo, filename=file_name)
            self.logger.info(f"Model downlaoded from Huggingface Hub")
            return model_path
        except Exception as e:
            self.logger.error(f"function download_model : {e}", exc_info=True)

    def rag(self, new_data=False):
        try:
            self.logger.info(f"Loading the RAG model")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-l6-v2",  # Provide the pre-trained model's path
                model_kwargs={'device': self.device},  # Pass the model configuration options
                encode_kwargs={'normalize_embeddings': False}  # Pass the encoding options
            )
            self.logger.info(f"Creating the FAISS db vector")
            if new_data:
                loader = DirectoryLoader(path=self.path, glob="**/**/*.csv", use_multithreading=True, loader_cls=CSVLoader)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
                all_splits = text_splitter.split_documents(data)
                db = FAISS.from_documents(all_splits, embeddings)
            else:
                DB_FAISS_PATH = 'db_faiss'
                db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            llm = LlamaCpp(model_path=self.rag_modelpath, n_ctx=5000, n_gpu_layers=1, n_batch=512,
                           f16_kv=True, callback_manager=callback_manager, verbose=True)
            prompt = ChatPromptTemplate(
                input_variables=['context', 'question'],
                messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'],
                                                                           template=r"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\If you don't know the answer, just say that you don't know.\\nQuestion: {question} \nContext: {context} \nAnswer:"""))])
            retriever = db.as_retriever(search_type=self.search_type, search_kwargs={"score_threshold": 0.25})

            qachain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever,
                                                  return_source_documents=True)
            # response = qachain.invoke({"query": self.query})
            # docs = retriever.get_relevant_documents(self.query)
            # qachain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
            # chain({"input_documents": docs, "question": self.query}, return_only_outputs=False)
            self.logger.info(f"Model configured successfully")
            return qachain#, retriever
        except Exception as e:
            self.logger.error(f"function rag: {e}", exc_info=True)
            return e, e