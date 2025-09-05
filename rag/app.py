import streamlit as st
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
import re
import uuid

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="RAG Document Analysis",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .result-container {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 1px solid #e6e9ef;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def clean_filename(filename):
    """Remove "(number)" pattern from a filename"""
    new_filename = re.sub(r'\s\(\d+\)', '', filename)
    return new_filename

# Pydantic model for structured output
class ExtractedInfo(BaseModel):
    paper_title: str = Field(..., description="The title of the paper")
    paper_summary: str = Field(..., description="A brief summary of the paper")
    key_findings: str = Field(..., description="Key findings from the paper")
    sources_used: list[str] = Field(..., description="List of sources or context sections referenced")

@st.cache_resource
def get_embedding_function():
    """Get embedding function with caching"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please set your OPENAI_API_KEY in the environment variables or .env file")
        st.stop()
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=api_key,
        model="text-embedding-3-small"
    )
    return embeddings

@st.cache_resource
def get_chat_model():
    """Get chat model with caching"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Please set your OPENAI_API_KEY in the environment variables or .env file")
        st.stop()
    
    chat_model = ChatOpenAI(
        openai_api_key=api_key,
        model="gpt-4o-mini"
    )
    return chat_model

def create_vectorstore(chunks, embedding_function, file_name, vector_store_path="vectorstore_streamlit"):
    """Create vector store from document chunks"""
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
        ids=unique_ids,
        collection_name=clean_filename(file_name),
        embedding=embedding_function, 
        persist_directory=vector_store_path
    )
    return vectorstore

def process_pdf(uploaded_file, embedding_function):
    """Process uploaded PDF file"""
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(tmp_file_path)
    data = loader.load()
    
    # Clean up temporary file
    os.unlink(tmp_file_path)

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(data)
    
    # Create vector store
    file_name = os.path.splitext(uploaded_file.name)[0]
    vectorstore = create_vectorstore(chunks, embedding_function, file_name)
    
    return vectorstore, len(chunks)

def format_docs(docs):
    """Format documents for context"""
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

def create_rag_chain(vectorstore, chat_model):
    """Create RAG chain"""
    retriever = vectorstore.as_retriever(search_type="similarity")
    
    PROMPT_TEMPLATE = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer
    the question. If you don't know the answer, say that you
    don't know. DON'T MAKE UP ANYTHING.

    When providing information, also keep track of which parts of the context
    you used to derive your answers so you can reference the sources.

    {context}

    ---

    Answer the question based on the above context: {question}

    Also list the specific 
    parts of the context that were most relevant for your answers.
    """
    
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    
    rag_chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | retriever | format_docs,
            "question": RunnableLambda(lambda x: x["question"])
        }
        | prompt_template
        | chat_model.with_structured_output(ExtractedInfo, strict=True)
    )
    
    return rag_chain

# Main Streamlit App
def main():
    # Header
    st.markdown('<h1 class="main-header">📄 RAG Document Analysis System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Key check
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            st.success("✅ OpenAI API Key loaded")
        else:
            st.error("❌ OpenAI API Key not found")
            st.info("Please set OPENAI_API_KEY in your environment variables or .env file")
        
        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown("""
        1. Upload a PDF document
        2. Wait for processing to complete
        3. Ask ANY question about the document
        4. Get personalized answers that match your question style
        
        **Try asking:**
        - "Explain this like I'm 10"
        - "What are the main points?"
        - "Give me technical details"
        - "Summarize in 3 sentences"
        """)

    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF document to analyze"
        )
    
    with col2:
        st.header("❓ Ask Questions")
        question = st.text_area(
            "Enter your question:",
            value="give me the summary title and key findings of the document",
            height=100,
            help="Ask specific questions about the uploaded document"
        )

    if uploaded_file is not None:
        # Initialize components
        embedding_function = get_embedding_function()
        chat_model = get_chat_model()
        
        # Process PDF
        with st.spinner("Processing PDF... This may take a moment."):
            try:
                vectorstore, chunk_count = process_pdf(uploaded_file, embedding_function)
                st.success(f"✅ PDF processed successfully! Created {chunk_count} text chunks.")
                
                # Document info
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.info(f"📄 **File:** {uploaded_file.name} | **Size:** {len(uploaded_file.getvalue())} bytes | **Chunks:** {chunk_count}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                return

        # Question answering
        if st.button("🔍 Analyze Document", type="primary"):
            if question.strip():
                with st.spinner("Analyzing document..."):
                    try:
                        # Create RAG chain
                        rag_chain = create_rag_chain(vectorstore, chat_model)
                        
                        # Get response
                        response = rag_chain.invoke({"question": question})
                        
                        # Display results
                        st.markdown('<div class="result-container">', unsafe_allow_html=True)
                        st.header("📊 Analysis Results")
                        
                        # Main results
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.subheader("📋 Paper Title")
                            st.write(response.paper_title)
                            
                            st.subheader("📝 Summary")
                            st.write(response.paper_summary)
                            
                            st.subheader("🔍 Key Findings")
                            st.write(response.key_findings)
                        
                        with col2:
                            st.subheader("📚 Sources Referenced")
                            if response.sources_used:
                                for i, source in enumerate(response.sources_used, 1):
                                    st.write(f"{i}. {source}")
                            else:
                                st.write("No specific sources listed")
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Raw response (expandable)
                        with st.expander("🔧 Raw Response Data"):
                            st.json({
                                "paper_title": response.paper_title,
                                "paper_summary": response.paper_summary,
                                "key_findings": response.key_findings,
                                "sources_used": response.sources_used
                            })
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {str(e)}")
            else:
                st.warning("⚠️ Please enter a question before analyzing.")
    
    else:
        st.info("👆 Please upload a PDF file to get started.")

if __name__ == "__main__":
    main()