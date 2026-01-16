import sys
import os
import shutil
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from resource_loader import ResourceLoader, HTML_DIR
from utils import FAISSVectorStore, VietnameseEmbedder

# Mock Embedder and Store to avoid heavy loading
class MockEmbedder:
    def embed_documents(self, texts):
        return [[0.1] * 768 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 768

class MockStore:
    def __init__(self):
        self.embedder = MockEmbedder()
        self.docs = []
    def add_documents(self, docs):
        self.docs.extend(docs)
    def add_documents_with_embeddings(self, docs, embs):
        self.docs.extend(docs)

def test_html_loader():
    print("Testing HTML Resource Loader...")
    
    # Setup
    loader = ResourceLoader()
    loader.set_vector_store(MockStore())
    
    # Create dummy HTML
    dummy_html = "test_html_resource.html"
    with open(dummy_html, "w", encoding="utf-8") as f:
        f.write("<html><body><h1>Test Title</h1><table><tr><td>Col1</td><td>Col2</td></tr></table></body></html>")
        
    try:
        # Test Add
        print(f"Adding {dummy_html}...")
        loader.add_html(dummy_html, "uploaded_test.html")
        
        target_path = HTML_DIR / "uploaded_test.html"
        if target_path.exists():
            print("SUCCESS: File copied to HTML_DIR")
        else:
            print("FAIL: File not copied")
            
        if "uploaded_test.html" in loader.loaded_resources:
             print("SUCCESS: Resource marked as loaded")
        else:
             print("FAIL: Resource not in loaded_resources")
             
        # Test Get
        res = loader.get_resources()
        found = any(r["name"] == "uploaded_test.html" and r["type"] == "html" for r in res)
        if found:
            print("SUCCESS: Resource listed in get_resources()")
        else:
            print("FAIL: Resource not listed")

        # Test Delete
        print("Deleting resource...")
        loader.delete_resource("uploaded_test.html")
        
        if not target_path.exists():
             print("SUCCESS: File deleted from HTML_DIR")
        else:
             print("FAIL: File still exists")
             
        if "uploaded_test.html" not in loader.loaded_resources:
             print("SUCCESS: Resource removed from loaded_resources")
        else:
             print("FAIL: Resource still in loaded_resources")

    except Exception as e:
        print(f"EXCEPTION: {e}")
    finally:
        # Cleanup local dummy
        if os.path.exists(dummy_html):
            os.remove(dummy_html)
        # Cleanup target if test failed
        t = HTML_DIR / "uploaded_test.html"
        if t.exists():
            os.remove(t)

if __name__ == "__main__":
    test_html_loader()
