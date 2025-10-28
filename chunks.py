import pickle
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class TextAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text = None
        self.load_text()
    
    def load_text(self):
        """Load text with proper UTF-8 encoding"""
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            self.text = f.read()
    
    def get_statistics(self):
        """Get text statistics"""
        stats = {
            "characters": len(self.text),
            "words": len(self.text.split()),
            "lines": len(self.text.splitlines())
        }
        return stats
    
    def print_statistics(self):
        """Print formatted statistics"""
        stats = self.get_statistics()
        print("Characters:", stats["characters"])
        print("Words:", stats["words"])
        print("Lines:", stats["lines"])
    
    def get_sample(self, start=5000, end=5200):
        """Get text sample from specified range"""
        return self.text[start:end]


class ChunkProcessor:
    def __init__(self, chunk_size=1000, chunk_overlap=150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_documents(self, file_path):
        """Load documents with explicit UTF-8 encoding"""
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()
    
    def create_chunks(self, documents):
        """Split documents into chunks"""
        return self.text_splitter.split_documents(documents)
    
    def save_chunks(self, chunks, output_path):
        """Save chunks to pickle file"""
        with open(output_path, "wb") as f:
            pickle.dump(chunks, f)
        print(f"✅ Chunks saved successfully to: {output_path}")
    
    def preview_chunks(self, chunks, num_chunks=3, preview_length=500):
        """Preview first n chunks"""
        for i, chunk in enumerate(chunks[:num_chunks]):
            print(f"\n--- Chunk {i+1} ---\n")
            print(chunk.page_content[:preview_length])


def main():
    # File paths
    cleaned_file_path = "output/python_docs_cleaned.txt"
    chunks_output_path = "output/python_docs_chunks.pkl"
    
    # Analyze text
    print("📊 TEXT ANALYSIS")
    analyzer = TextAnalyzer(cleaned_file_path)
    analyzer.print_statistics()
    print("\n📄 Sample snippet:\n", analyzer.get_sample())
    
    # Process chunks
    print("\n🔪 CHUNK PROCESSING")
    processor = ChunkProcessor(chunk_size=1000, chunk_overlap=150)
    
    documents = processor.load_documents(cleaned_file_path)
    print(f"Loaded {len(documents)} document(s)")
    
    chunks = processor.create_chunks(documents)
    print(f"Number of chunks created: {len(chunks)}")
    
    # Save chunks
    processor.save_chunks(chunks, chunks_output_path)
    
    # Preview chunks
    print("\n👀 CHUNK PREVIEW")
    processor.preview_chunks(chunks)
    
    return chunks

if __name__ == "__main__":
    chunks = main()