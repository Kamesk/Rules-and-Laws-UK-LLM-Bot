
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import numpy as np
from langchain_community.vectorstores import FAISS

huggingface_embeddings=HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",      #sentence-transformers/all-MiniLM-l6-v2
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings':True})

embeddings = [huggingface_embeddings.embed_query(text) for text in texts]
embeddings_array = np.array(embeddings)
print(embeddings_array)

vectorstore = FAISS.from_documents(final_documents, huggingface_embeddings)