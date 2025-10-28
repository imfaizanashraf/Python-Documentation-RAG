from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle
import os
import gc
import time

class EmbeddingGenerator:
    def __init__(self, chunks_path, output_dir="vector_store", model_name="sentence-transformers/all-MiniLM-L12-v2"):
        self.chunks_path = chunks_path
        self.output_dir = output_dir
        self.model_name = model_name
        self.chunks = None
        self.embeddings = None
        self.vectorstore = None
        
    def load_chunks(self):
        """Load chunks from pickle file"""
        with open(self.chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        print(f"📚 Total chunks to embed: {len(self.chunks)}")
        return self.chunks
    
    def initialize_embeddings(self):
        """Initialize the embedding model"""
        print(f"🔄 Loading embedding model: {self.model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
        print(f"✅ Embedding model loaded: {self.model_name}")
        return self.embeddings
    
    def create_vector_store(self, batch_size=4000):
        """Create FAISS vector store with batch processing and auto-saving"""
        if not self.chunks:
            raise ValueError("Chunks not loaded. Call load_chunks() first.")
        if not self.embeddings:
            raise ValueError("Embeddings not initialized. Call initialize_embeddings() first.")
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.vectorstore = None
        
        total_batches = (len(self.chunks) + batch_size - 1) // batch_size
        start_time = time.time()
        
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"\n🚀 Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            batch_start = time.time()
            
            batch_vs = FAISS.from_documents(batch, self.embeddings)
            
            if self.vectorstore is None:
                self.vectorstore = batch_vs
            else:
                self.vectorstore.merge_from(batch_vs)
            
            print(f"✅ Batch {batch_num} merged successfully. Saving progress...")
            self.vectorstore.save_local(self.output_dir)
            
            # Clean up memory
            del batch, batch_vs
            gc.collect()
            
            elapsed = time.time() - batch_start
            print(f"⏱️ Batch {batch_num} done in {elapsed:.2f} seconds.")
        
        total_time = (time.time() - start_time) / 60
        print(f"\n🎉 All batches processed in {total_time:.1f} minutes.")
        return self.vectorstore
    
    def get_vector_store_info(self):
        """Get information about the vector store"""
        if not self.vectorstore:
            return "Vector store not created"
        
        info = {
            "total_chunks": len(self.chunks),
            "embedding_model": self.model_name,
            "save_location": self.output_dir
        }
        return info

def main():
    CHUNKS_PATH = "output/python_docs_chunks.pkl"
    OUTPUT_DIR = "vector_store"
    
    generator = EmbeddingGenerator(CHUNKS_PATH, OUTPUT_DIR)
    
    # Step 1: Load chunks
    generator.load_chunks()
    
    # Step 2: Initialize embeddings (faster model)
    generator.initialize_embeddings()
    
    # Step 3: Create and save vector store batch-wise
    generator.create_vector_store(batch_size=4000)
    
    # Step 4: Display summary
    info = generator.get_vector_store_info()
    print(f"\n📊 Vector Store Information:")
    for k, v in info.items():
        print(f"   {k}: {v}")

if __name__ == "__main__":
    embedding_generator = main()
