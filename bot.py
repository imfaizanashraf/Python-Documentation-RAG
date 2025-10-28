import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class RAGChatBot:
    def __init__(self, vector_store_path, model_name=None):
        self.vector_store_path = vector_store_path
        self.model_name = model_name
        self.vectorstore = None
        self.client = None
        self.available_models = [
            "mistralai/Mistral-7B-Instruct-v0.3",
            "google/gemma-2-2b-it",
            "microsoft/DialoGPT-medium",
            "HuggingFaceH4/zephyr-7b-beta",
            "meta-llama/Llama-2-7b-chat-hf"
        ]
        
    def initialize_components(self):
        """Initialize both vector store and LLM client"""
        st.session_state.status = "🔄 Initializing RAG ChatBot..."
        
        # Load vector store
        try:
            self.vectorstore = FAISS.load_local(
                self.vector_store_path, 
                HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2"),
                allow_dangerous_deserialization=True
            )
            
            # Test the vector store by doing a sample search
            test_results = self.vectorstore.similarity_search("python", k=1)
            if not test_results:
                st.error("⚠️ Vector store loaded but no documents found in test search")
                return False
                
        except Exception as e:
            st.error(f"❌ Error loading vector store: {e}")
            return False
        
        # Initialize OpenAI-compatible client
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=os.environ["HF_TOKEN"]
        )
        
        # Auto-detect available model
        if not self.model_name:
            self.model_name = self.detect_available_model()
        else:
            if not self.test_model_availability(self.model_name):
                self.model_name = self.detect_available_model()
        
        st.session_state.status = f"✅ Ready! Using model: {self.model_name}"
        return True
        
    def test_model_availability(self, model_name):
        """Test if a model is available"""
        try:
            self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            return True
        except:
            return False
    
    def detect_available_model(self):
        """Detect which models are available from the list"""
        for model in self.available_models:
            try:
                if self.test_model_availability(model):
                    return model
            except:
                continue
        
        # If no models from the list work, try fallback models
        fallback_models = [
            "microsoft/DialoGPT-large",
            "HuggingFaceH4/zephyr-7b-beta", 
            "google/flan-t5-large"
        ]
        for model in fallback_models:
            if self.test_model_availability(model):
                return model
                
        raise Exception("No available models found.")
    
    def search_relevant_context(self, query, k=4):
        """Search for relevant context from vector store"""
        if not self.vectorstore:
            raise ValueError("Vector store not loaded")
            
        docs = self.vectorstore.similarity_search(query, k=k)
        
        if not docs:
            return "No relevant information found in the knowledge base."
            
        context = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
        return context
    
    def build_messages(self, query, context):
        """Build the messages array for chat completion"""
        system_message = {
            "role": "system",
            "content": f"""You are a helpful AI assistant that answers questions STRICTLY based on the provided context.

IMPORTANT INSTRUCTIONS:
1. You MUST use the information from the provided context to answer the question
2. If the context contains relevant information, base your answer ONLY on that information
3. If the context does not contain the answer, say "I couldn't find specific information about this in my knowledge base."
4. Do not make up information or use external knowledge

CONTEXT:
{context}

QUESTION: {query}

Please provide a helpful answer based strictly on the context above:"""
        }
        
        messages = [system_message]
        return messages
    
    def generate_response(self, query):
        """Generate response using RAG"""
        if not self.client or not self.vectorstore:
            raise ValueError("Components not initialized")
        
        # Get relevant context
        context = self.search_relevant_context(query)
        
        # Build messages array
        messages = self.build_messages(query, context)
        
        try:
            # Generate response using OpenAI-compatible client
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=800,
                temperature=0.1,
                top_p=0.9,
                stream=False
            )
            
            response = completion.choices[0].message.content
            return response, context
            
        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            return error_msg, None

def initialize_chatbot():
    """Initialize the chatbot and store in session state"""
    VECTOR_STORE_PATH = "vector_store"
    MODEL_NAME = None
    
    if not os.path.exists(VECTOR_STORE_PATH):
        st.error(f"❌ Vector store directory '{VECTOR_STORE_PATH}' not found!")
        st.stop()
    
    if "HF_TOKEN" not in os.environ:
        st.error("❌ HF_TOKEN not found in environment variables!")
        st.stop()
    
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = RAGChatBot(VECTOR_STORE_PATH, MODEL_NAME)
        success = st.session_state.chatbot.initialize_components()
        if not success:
            st.error("❌ Failed to initialize chatbot components")
            st.stop()

def main():
    st.set_page_config(
        page_title="RAG ChatBot",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_sources" not in st.session_state:
        st.session_state.show_sources = False
    if "last_context" not in st.session_state:
        st.session_state.last_context = None
    
    # Header
    st.title("🤖 RAG ChatBot")
    st.markdown("Chat with your documents using Retrieval Augmented Generation")
    
    # Initialize chatbot
    initialize_chatbot()
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Model info
        if hasattr(st.session_state.chatbot, 'model_name'):
            st.info(f"**Model:** {st.session_state.chatbot.model_name}")
        
        # Conversation controls
        st.subheader("Conversation")
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_context = None
            st.rerun()
        
        # Toggle for showing sources
        st.session_state.show_sources = st.toggle("Show Context Sources", value=False)
        
        # Test vector store
        if st.button("🔍 Test Vector Store", use_container_width=True):
            with st.spinner("Testing vector store..."):
                try:
                    test_query = "python"
                    docs = st.session_state.chatbot.vectorstore.similarity_search(test_query, k=2)
                    if docs:
                        st.success(f"✅ Found {len(docs)} documents for test query")
                        for i, doc in enumerate(docs):
                            with st.expander(f"Document {i+1}"):
                                st.text(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                    else:
                        st.error("❌ No documents found!")
                except Exception as e:
                    st.error(f"Error testing vector store: {e}")
        
        # Status
        if hasattr(st.session_state, 'status'):
            st.info(st.session_state.status)
    
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show context sources if enabled and available
                if (message["role"] == "assistant" and 
                    st.session_state.show_sources and 
                    message.get("context")):
                    with st.expander("🔍 Context Sources"):
                        st.text(message["context"])
    
    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                try:
                    response, context = st.session_state.chatbot.generate_response(prompt)
                    st.markdown(response)
                    
                    # Store context for this response
                    message_data = {"role": "assistant", "content": response}
                    if context:
                        message_data["context"] = context
                    
                    st.session_state.messages.append(message_data)
                    st.session_state.last_context = context
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("Built with Streamlit & LangChain")
    with col2:
        st.caption("Powered by Hugging Face")
    with col3:
        st.caption("RAG Architecture")

if __name__ == "__main__":
    main()