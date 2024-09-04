from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.llms import CTransformers
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
import numpy as np
import re
from src.helper import *

path_to_pdfs = (r"D:\LLM\projects\Questionme\artifacts")
loader = PyPDFDirectoryLoader(path_to_pdfs)
pdf_documents = loader.load()

def cleanup(document):
    document = document.replace('\n', ' ')
    document = document.replace('-', '')
    document = ' '.join(document.split())
    document = re.sub(r'\n+', ' ', document)
    return document

cleaned_documents = [cleanup(doc.page_content) for doc in pdf_documents]
cleaned_document_objects = [Document(page_content=cleaned_text, metadata=doc.metadata) 
                             for cleaned_text, doc in zip(cleaned_documents, pdf_documents)]
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
final_documents=text_splitter.split_documents(cleaned_document_objects)
texts = [doc.page_content for doc in final_documents]
text = texts

llm = CTransformers(
    model=r"D:\LLM\models\llama-2-7b-chat.ggmlv3.q8_0.bin",
    model_type="llama",
    config={'max_new_tokens': 300, 'temperature': 0.03,'context_length': 1000}
)

huggingface_embeddings=HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True})

embeddings = [huggingface_embeddings.embed_query(text) for text in texts]
embeddings_array = np.array(embeddings)

vectorstore = FAISS.from_documents(final_documents, huggingface_embeddings)

qa_prompt = PromptTemplate(template=template, input_variables=['context', 'question'])

retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=retriever,
    return_source_documents=False,
    chain_type_kwargs={'prompt': qa_prompt}
)

user_input = "Who sets the level of FPN's"

result=chain({'query':user_input})
print(f"Answer:{result['result']}")