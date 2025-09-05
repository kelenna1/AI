# 📄 RAG Document Analysis System

A powerful **Retrieval-Augmented Generation (RAG)** application built with **Streamlit** that allows you to upload PDF documents and ask natural language questions about their content. The system uses advanced AI to provide intelligent, context-aware responses with source citations.

## 🌟 Features

- **📤 PDF Upload**: Drag and drop PDF documents for analysis
- **🤖 Intelligent Q&A**: Ask questions in natural language and get contextual answers
- **🎯 Adaptive Responses**: The AI adapts its language and style based on your question
- **📚 Source Citations**: Every answer includes references to specific document sections
- **🔍 Confidence Scoring**: Get confidence levels (High/Medium/Low) for each response
- **⚡ Real-time Processing**: Fast document processing with progress indicators
- **🎨 Modern UI**: Clean, responsive interface with professional styling
- **💾 Vector Storage**: Persistent vector database for efficient document retrieval

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API Key

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd rag-document-analysis
```

2. **Create virtual environment**
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

5. **Run the application**
```bash
streamlit run streamlit_app.py
```

6. **Open in browser**
Navigate to `http://localhost:8501`

## 📦 Dependencies

```txt
streamlit
langchain
langchain-community
langchain-openai
langchain-chroma
pydantic
python-dotenv
pypdf
chromadb
```

## 🎯 Usage Examples

### Basic Questions
- *"What is this document about?"*
- *"Summarize the main points"*
- *"What are the key findings?"*

### Adaptive Language
- *"Explain this like I'm 10 years old"* → Simple, kid-friendly explanation
- *"Give me the technical details"* → In-depth technical analysis
- *"Summarize in 3 bullet points"* → Concise bullet-point format

### Specific Queries
- *"What methodology was used?"*
- *"What are the limitations of this study?"*
- *"How reliable is this research?"*

## 🏗️ Architecture

The system uses a **RAG (Retrieval-Augmented Generation)** pipeline:

1. **Document Processing**: PDF → Text chunks → Vector embeddings
2. **Storage**: ChromaDB vector database for efficient similarity search
3. **Retrieval**: Semantic search to find relevant document sections
4. **Generation**: OpenAI GPT-4 generates contextual responses
5. **Output**: Structured responses with source citations

### Key Components

- **LangChain**: RAG pipeline orchestration
- **ChromaDB**: Vector database for document storage
- **OpenAI Embeddings**: Text-to-vector conversion
- **OpenAI GPT-4**: Response generation
- **Streamlit**: Web interface
- **Pydantic**: Structured output validation

## 📊 Technical Features

### Document Processing
- **Text Splitting**: Recursive character splitting with overlap
- **Chunk Size**: 1000 characters with 200 character overlap
- **Deduplication**: UUID-based duplicate chunk removal
- **Metadata Preservation**: Page numbers and source tracking

### Vector Search
- **Embedding Model**: OpenAI `text-embedding-3-small`
- **Search Type**: Semantic similarity search
- **Storage**: Persistent ChromaDB with collection management
- **Retrieval**: Top-k relevant chunks for context

### Response Generation
- **Model**: OpenAI GPT-4-mini for cost-effective performance
- **Structured Output**: Pydantic models for consistent formatting
- **Confidence Scoring**: Built-in confidence assessment
- **Source Attribution**: Automatic citation of relevant sections

## 🛠️ Configuration

### Environment Variables
```env
OPENAI_API_KEY=your_api_key_here
```

### Customizable Parameters
- **Chunk Size**: Modify `chunk_size` in text splitter (default: 1000)
- **Overlap**: Adjust `chunk_overlap` for context preservation (default: 200)
- **Model Selection**: Change OpenAI models in configuration
- **Vector Store Path**: Customize storage location

## 🎨 User Interface

### Layout
- **Sidebar**: Configuration, instructions, and API status
- **Main Area**: File upload and question interface
- **Results**: Structured response display with confidence indicators
- **Expandable Sections**: Raw data view for debugging

### Styling
- **Custom CSS**: Professional appearance with branded colors
- **Responsive Design**: Works on desktop and mobile devices
- **Progress Indicators**: Real-time processing feedback
- **Error Handling**: User-friendly error messages

## 🔧 Development

### Project Structure
```
rag-document-analysis/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── .env                     # Environment variables (create this)
├── .env.example            # Environment template
├── vectorstore_streamlit/  # ChromaDB storage (auto-created)
└── README.md              # This file
```

### Running in Development
```bash
streamlit run streamlit_app.py --logger.level=debug
```

### Testing
Upload sample PDFs and test various question types:
- Academic papers
- Business reports  
- Technical documentation
- Research studies

## 🚀 Deployment

### Streamlit Cloud
1. Push to GitHub
2. Connect to Streamlit Cloud
3. Add `OPENAI_API_KEY` to secrets
4. Deploy automatically

### Local Production
```bash
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🐛 Troubleshooting

### Common Issues

**API Key Error**
```
❌ OpenAI API Key not found
```
Solution: Ensure `.env` file exists with valid `OPENAI_API_KEY`

**PDF Processing Error**
```
❌ Error processing PDF
```
Solution: Verify PDF is not corrupted and contains readable text

**Memory Issues**
```
Vector store creation failed
```
Solution: Try smaller PDF files or increase system memory

### Performance Tips
- Use smaller PDFs (< 50 pages) for faster processing
- Clear vector store occasionally to free disk space
- Restart app if experiencing memory issues

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** - RAG framework
- **OpenAI** - Language models and embeddings
- **Streamlit** - Web application framework
- **ChromaDB** - Vector database

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Open an issue on GitHub
3. Review LangChain documentation

---

**⭐ Star this repo if you found it helpful!**