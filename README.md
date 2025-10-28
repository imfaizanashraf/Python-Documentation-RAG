# Python Documentation Chatbot

This is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about Python documentation using embeddings and a language model. The system leverages state-of-the-art NLP techniques to provide accurate and contextually relevant answers based on official Python documentation.

## 🌟 Features

- **Knowledge Base**: Uses comprehensive Python documentation as the knowledge source
- **Semantic Search**: Implements FAISS vector store for efficient similarity search
- **Advanced NLP**: Uses HuggingFace models for embeddings and language generation
- **User Interface**: Provides an intuitive Streamlit web interface for chatting
- **Modular Design**: Clean architecture with separate modules for data processing, chunking, and embedding

## 🚀 Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd python_rag_chatbot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure HuggingFace API token**:
   - Visit [HuggingFace](https://huggingface.co/) and create an account
   - Generate an API token from your profile settings
   - Add your token to the `.env` file:
     ```
     HUGGINGFACE_API_TOKEN=your_token_here
     ```

4. **Run the application**:
   ```bash
   streamlit run bot.py
   ```

5. **Access the chatbot**:
   Open your browser to `http://localhost:8501` (Streamlit's default)

## 🧠 How It Works

1. **Data Processing**: The system processes Python documentation files and creates text chunks
2. **Embedding Generation**: Chunks are converted to embeddings using sentence-transformers
3. **Vector Storage**: Embeddings are stored in a FAISS vector database for efficient retrieval
4. **Question Answering**: When you ask a question:
   - Your question is converted to an embedding
   - FAISS finds the most relevant documentation chunks
   - Chunks and question are sent to a language model
   - The model's response is returned to you

## 📁 Project Structure

```
python_rag_chatbot/
├── bot.py                 # Streamlit chatbot interface
├── data_processor.py      # Extracts and cleans text from documentation
├── chunks.py              # Splits text into chunks for processing
├── embeddings.py          # Creates and stores embeddings in FAISS
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in repo)
├── data/                  # Source documentation files
├── output/                # Processed text files
└── vector_store/          # FAISS vector database
```

## 🛠️ Technologies Used

- **LangChain**: Orchestration framework for RAG applications
- **FAISS**: Facebook AI Similarity Search for vector storage and retrieval
- **HuggingFace**: State-of-the-art transformer models for embeddings and generation
- **Streamlit**: Web framework for creating interactive data applications
- **Sentence Transformers**: For generating high-quality sentence embeddings

## 🔧 Requirements

- Python 3.8 or higher
- HuggingFace account with API token
- Minimum 4GB RAM (8GB recommended)
- Internet connection for model downloads

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

For questions or feedback, please open an issue on GitHub.