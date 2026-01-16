import sys
import os
import argparse
from typing import List
from langchain_core.documents import Document

# Add src to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

try:
    from crawler import crawl_url
except ImportError:
    print("Error: Could not import 'crawler'. Make sure you are running this from the project root or 'test' folder.")
    sys.exit(1)

# Force UTF-8 encoding for Windows Console
sys.stdout.reconfigure(encoding='utf-8')

def print_separator(char='-', length=60):
    print(char * length)

def visualize_chunk(doc: Document, index: int, total: int):
    """Pretty prints a single document chunk."""
    print_separator('=')
    print(f"📄 CHUNK {index}/{total}")
    print_separator('=')
    
    # Metadata Section
    print(f"📌 Metadata:")
    for key, value in doc.metadata.items():
        print(f"   - {key}: {value}")
    
    print_separator()
    
    # Content Section
    content = doc.page_content
    word_count = len(content.split())
    
    print(f"📝 Content ({word_count} words):")
    print(content)
    print_separator()
    print("\n")
    return content

def save_chunks_to_file(chunks: List[Document], filename: str):
    """Saves all chunks to a text file."""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"CRAWL EXPORT: {len(chunks)} chunks\n")
            f.write("="*60 + "\n\n")
            
            for i, doc in enumerate(chunks, 1):
                f.write(f"📄 CHUNK {i}/{len(chunks)}\n")
                f.write("-"*60 + "\n")
                f.write("📌 Metadata:\n")
                for key, value in doc.metadata.items():
                    f.write(f"   - {key}: {value}\n")
                f.write("\n📝 Content:\n")
                f.write(doc.page_content)
                f.write("\n\n" + "="*60 + "\n\n")
        print(f"\n💾 Output saved to: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"❌ Failed to save file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Test Web Crawler and Chunking Logic")
    parser.add_argument("url", nargs="?", help="URL or Local File Path to crawl", default=None)
    parser.add_argument("-o", "--output", help="Save output to file (e.g. chunks.txt)", default=None)
    args = parser.parse_args()

    target_url = args.url

    # Interactive mode if no argument provided
    if not target_url:
        print("\n🌐 Web Crawler Verification Tool")
        print("Enter a URL (e.g., https://example.com) or a Local File Path.")
        target_url = input("Target: ").strip()

    # Normalize path if it looks like a file
    if os.path.exists(target_url):
        target_url = os.path.abspath(target_url)
        print(f"\n📂 Mode: Local File Crawl")
    else:
        print(f"\n🌐 Mode: Network Crawl")

    print(f"🎯 Target: {target_url}\n")
    
    try:
        print("⏳ Processing... (This may take a moment)")
        chunks = crawl_url(target_url)
        
        if not chunks:
            print("\n❌ No chunks retrieved.")
            print("Possible reasons:")
            print("  1. WAF Blocked (468/403) -> Try saving as HTML and using local path.")
            print("  2. Page is empty or heavily Javascript-dependent.")
            print("  3. Content filtered out as spam.")
            return

        print(f"\n✅ Successfully retrieved {len(chunks)} chunks!\n")
        
        for i, chunk in enumerate(chunks, 1):
            visualize_chunk(chunk, i, len(chunks))
            
        if args.output:
            save_chunks_to_file(chunks, args.output)
            
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
