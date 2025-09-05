from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import os
from dotenv import load_dotenv
import tempfile
import streamlit as st
import pandas as pd
import re

# Load environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

def clean_filename(filename):
    """
    Remove "(number)" pattern from a filename 
    (because this could cause error when used as collection name when creating Chroma database).

    Parameters:
        filename (str): The filename to clean

    Returns:
        str: The cleaned filename
    """
    # Regular expression to find "(number)" pattern
    new_filename = re.sub(r'\s\(\d+\)', '', filename)
    return new_filename

# Chat model
chat_model = ChatOpenAI(
    openai_api_key=api_key,
    model="gpt-4o-mini"  # You can also use "gpt-4o-mini", "gpt-4.1", etc.
)

# Load PDF
loader = PyPDFLoader("data/dang.pdf")
data = loader.load()

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunk = text_splitter.split_documents(data)

# Embeddings function
def get_embedding_function():
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-3-small"  # or "text-embedding-3-large"
    )
    return embeddings

embedding_function = get_embedding_function()

import uuid
# Create Chroma vector store
def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="vector_store"):
    seen_ids = set()
    unique_chunks = []
    unique_ids = []

    for doc in chunks:
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.page_content))
        if doc_id not in seen_ids:
            seen_ids.add(doc_id)
            unique_chunks.append(doc)
            unique_ids.append(doc_id)

    vectorstore = Chroma.from_documents(
        documents=unique_chunks, 
        ids=unique_ids,  # IDs now properly aligned with chunks
        collection_name=clean_filename(file_name),  # Cleaned filename for collection name
        embedding=embedding_function, 
        persist_directory=vector_store_path
    )
    return vectorstore

vectorstore = create_vectorstore(
    chunks= chunk,
    embedding_function= embedding_function,
    vector_store_path= "vectorstore_chroma",
    file_name=os.path.splitext(os.path.basename("data/dang.pdf"))[0])

#load database
vectorstore = Chroma(
    collection_name=clean_filename(os.path.splitext(os.path.basename("data/dang.pdf"))[0]),
    persist_directory="vectorstore_chroma",
    embedding_function=embedding_function,
)

retriever = vectorstore.as_retriever(search_type="similarity") 
relevant_chunks= retriever.invoke("What is the main topic of the document?")



#Prompt template
PROMPT_TEMPLATE = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer
the question. If you don't know the answer, say that you
don't know. DON'T MAKE UP ANYTHING.

{context}

---

Answer the question based on the above context: {question}
"""

context_text= "\n\n---\n\n".join([doc.page_content for doc in relevant_chunks])

prompt_template= ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt= prompt_template.format_prompt(
    context=context_text,
    question= "What is the main topic of the document?"
)

def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

# Define structured output model
class ExtractedInfo(BaseModel):
    paper_title: str = Field(..., description="The title of the paper")
    paper_summary: str = Field(..., description="A brief summary of the paper")
    key_findings: str = Field(..., description="Key findings from the paper")
    sources_used: list[str] = Field(..., description="List of sources referenced")


rag_chain = (
    {
        "context": RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
        "question": RunnableLambda(lambda x: x["question"])
    }
    | prompt_template
    | chat_model.with_structured_output(ExtractedInfo, strict=True)
)

# Test the chain
# response = rag_chain.invoke({"question": "give me the summary title and key findings of the document"})
# print(response)

#Transform response into a dataframe
structured_response = rag_chain.invoke({"question": "give me the summary title and key findings of the document"})
df = pd.DataFrame([structured_response.model_dump()])
print(df)



