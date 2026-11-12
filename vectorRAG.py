from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()
#environment variables
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
# define and initialise LLM model
llm = ChatGroq(groq_api_key=groq_api_key, model="Llama3-8b-8192")
from langchain_huggingface import HuggingFaceEmbeddings
#all-MiniLM-L6-v2 is for sentence transformer model, it maps sentences and paragraphs to a 384 dimentional dense vector space
# and can be used for task like clustering or semantic search
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

documents = [
    Document(page_content="Khichdi is a popular Indian dish made with rice and lentils.", metadata={"source": "source1","type": "description"}),
    Document(page_content="To make khichdi, you need rice, lentils, water, and spices.", metadata={"source": "source2","type": "recipe"}),
    Document(page_content="Khichdi is often served with yogurt or pickle.", metadata={"source": "source3","type": "serving"}),
    Document(page_content="Khichdi is considered a comfort food in India.", metadata={"source": "source4","type": "cultural"})
]
##vector stores
from langchain_chroma import Chroma
vectorstore = Chroma.from_documents(documents, embeddings)
query1 = "What is khichdi?"
results = vectorstore.similarity_search(query1)
for result in results:
    print(result.page_content)
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
retriever_runnable = RunnableLambda(vectorstore.similarity_search).bind(k=1)
retriever_runnable.batch([query1, "How is khichdi served?"])
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

#integrate with chain
from langchain_core.prompts import ChatPromptTemplate
message = """
Answer the question using the provided context only.
{question}

Context:
{context}
"""
prompt = ChatPromptTemplate.from_messages([("human", message)])

# create RAG chain 
rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
response = rag_chain.invoke("tell me about khichdi")
print(response.content)
