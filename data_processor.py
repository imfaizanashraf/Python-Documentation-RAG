import os
import re
from bs4 import BeautifulSoup

class DataExtractor:
    def __init__(self, source_dir="data", output_dir="output"):
        self.source_dir = source_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_text_from_file(self, file_path, file_name):
        """Extract and clean text from a single file"""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            if file_name.endswith((".html", ".htm")):
                soup = BeautifulSoup(content, "html.parser")
                for tag in soup(["script", "style"]):
                    tag.extract()
                text = soup.get_text(separator="\n", strip=True)
            else:
                text = content

            return "\n".join([line.strip() for line in text.splitlines() if line.strip()])
        except Exception as e:
            print(f"⚠️ Skipping {file_name}: {e}")
            return None

    def extract_all_data(self):
        """Extract data from all files in source directory"""
        all_text = ""
        
        for root, _, files in os.walk(self.source_dir):
            for file in files:
                if file.endswith((".html", ".htm", ".txt")):
                    path = os.path.join(root, file)
                    text = self.extract_text_from_file(path, file)
                    
                    if text:
                        all_text += f"\n\n### FILE: {file} ###\n{text}"

        output_file = os.path.join(self.output_dir, "python_docs_text.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(all_text)
        
        print("✅ Extracted clean text saved to:", output_file)
        return output_file


class DataCleaner:
    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
    
    def clean_extracted_data(self, input_file):
        """Clean the extracted text data"""
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        noise_patterns = [
            r'\bTheme\b', r'\bAuto\b', r'\bLight\b', r'\bDark\b',
            r'\bNavigation\b', r'\bindex\b', r'\bmodules\b',
            r'\bPython\b', r'\bDocumentation\b', r'\b©\b',
            r'\bTable of Contents\b', r'\bQuick search\b'
        ]

        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        text = re.sub(r'[\|\»›«]+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        text = text.strip()

        output_file = os.path.join(self.output_dir, "python_docs_cleaned.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(text)

        print("✅ Cleaned text saved to:", output_file)
        return output_file


def main():
    # Initialize components
    extractor = DataExtractor(source_dir="data")
    cleaner = DataCleaner()
    
    # Execute pipeline
    extracted_file = extractor.extract_all_data()
    cleaned_file = cleaner.clean_extracted_data(extracted_file)
    
    print("🎯 Data processing completed!")
    return cleaned_file

if __name__ == "__main__":
    main()