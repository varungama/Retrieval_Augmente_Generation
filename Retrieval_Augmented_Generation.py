import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from llama_cpp import Llama
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
# from langchain.chains.question_answering import load_qa_chain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA


class Rag_model:
    def __init__(self):
        self.path = r'output_dfs'
        self.retriver_type = "single"
        self.rag_modelpath = os.path.join(os.getcwd(), r'Model\mistral-7b-instruct-v0.1.Q5_K_M.gguf')

    def rag(self):
        try:
            loader = DirectoryLoader(path=self.path, glob="**/**/*.csv", use_multithreading=True, loader_cls=CSVLoader)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            all_splits = text_splitter.split_documents(data)
            print(len(all_splits))
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-l6-v2",  # Provide the pre-trained model's path
                model_kwargs={'device': 'cpu'},  # Pass the model configuration options
                encode_kwargs={'normalize_embeddings': False}  # Pass the encoding options
            )
            # DB_FAISS_PATH = 'db_faiss'
            db = FAISS.from_documents(all_splits, embeddings)

            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            # Load model directly

            llm_online = Llama.from_pretrained(
                repo_id="TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                filename="mistral-7b-instruct-v0.1.Q2_K.gguf",
            )
            # llm = LlamaCpp(model_path=self.rag_modelpath, n_ctx=5000, n_gpu_layers=1, n_batch=512,
            #                f16_kv=True, callback_manager=callback_manager, verbose=True)
            # prompt = ChatPromptTemplate(
            #     input_variables=['context', 'question'],
            #     messages=[HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'],
            #                                                                template=r"""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.\If you don't know the answer, just say that you don't know.\\nQuestion: {question} \nContext: {context} \nAnswer:"""))])
            if self.retriver_type=="single":
                retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 8})
            elif self.retriver_type=="multiple":
                retriever = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm_online)

            qachain = RetrievalQA.from_chain_type(llm=llm_online, chain_type="stuff", retriever=retriever,
                                                  return_source_documents=True)
            # response = qachain.invoke({"query": self.query})
            # docs = retriever.get_relevant_documents(self.query)
            # chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
            # chain({"input_documents": docs, "question": self.query}, return_only_outputs=False)
            return qachain
        except Exception as e:
            print(e)
            return e