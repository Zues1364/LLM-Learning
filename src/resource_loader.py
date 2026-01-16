import os
import json
import logging
from pathlib import Path
from typing import List, Dict
import hashlib
import shutil

from utils import FAISSVectorStore, VietnameseEmbedder, process_pdf, load_embeddings_with_cache
from crawler import crawl_url
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RESOURCE_DIR = BASE_DIR / "data" / "resources"
PDF_DIR = RESOURCE_DIR / "pdfs"
HTML_DIR = RESOURCE_DIR / "html"
CONFIG_FILE = RESOURCE_DIR / "config.json"

# Ensure directories exist
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)
if not CONFIG_FILE.exists():
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump({"urls": []}, f)

class ResourceLoader:
    def __init__(self, vector_store: FAISSVectorStore = None):
        self.vector_store = vector_store
        self.loaded_resources = set()

    def set_vector_store(self, store: FAISSVectorStore):
        self.vector_store = store

    def _load_config(self) -> Dict:
        if not CONFIG_FILE.exists():
            return {"urls": []}
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {"urls": []}

    def _save_config(self, config: Dict):
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

    def load_resources(self):
        """
        Scans resource directory and config for PDFs and URLs, ingests them into VectorStore.
        """
        if not self.vector_store:
            logger.warning("[ResourceLoader] Vector store not set. Skipping load.")
            return

        logger.info("[ResourceLoader] strict loading global resources...")
        
        # 1. Load PDFs
        for pdf_file in PDF_DIR.glob("*.pdf"):
            file_id = pdf_file.name
            if file_id in self.loaded_resources:
                continue
            
            try:
                logger.info(f"[ResourceLoader] Ingesting PDF: {pdf_file.name}")
                embedder = self.vector_store.embedder
                
                docs = process_pdf(str(pdf_file))
                # Fix metadata to mark as resource
                for d in docs:
                    d.metadata["is_global_resource"] = True
                
                embeddings = load_embeddings_with_cache(str(pdf_file), embedder, docs)
                self.vector_store.add_documents_with_embeddings(docs, embeddings)
                self.loaded_resources.add(file_id)
            except Exception as e:
                logger.error(f"[ResourceLoader] Failed to load PDF {pdf_file.name}: {e}")

        # 2. Load HTMLs
        for html_file in HTML_DIR.glob("*.html"):
            file_id = html_file.name
            if file_id in self.loaded_resources:
                continue
            
            try:
                logger.info(f"[ResourceLoader] Ingesting HTML: {html_file.name}")
                # crawler supports local file paths
                docs = crawl_url(str(html_file))
                
                # Fix metadata
                for d in docs:
                    d.metadata["is_global_resource"] = True
                    # Use filename as ID. 
                    # Note: crawl_url might put 'file_id' in metadata, we override or ensure consistency
                    d.metadata["file_id"] = file_id 
                    # Also ensure 'file_name' is set for display
                    d.metadata["file_name"] = file_id
                
                if docs:
                    self.vector_store.add_documents(docs)
                    self.loaded_resources.add(file_id)
            except Exception as e:
                logger.error(f"[ResourceLoader] Failed to load HTML {html_file.name}: {e}")

        # 2. Load URLs
        config = self._load_config()
        urls = config.get("urls", [])
        for url_entry in urls:
            url = url_entry.get("url")
            if not url: continue
            
            url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
            pseudo_id = f"url_{url_hash}"
            
            if pseudo_id in self.loaded_resources:
                continue
                
            try:
                logger.info(f"[ResourceLoader] Crawling URL: {url}")
                docs = crawl_url(url)
                # Mark metadata
                for d in docs:
                    d.metadata["is_global_resource"] = True
                    d.metadata["file_id"] = pseudo_id # Ensure ID match
                
                if docs:
                    self.vector_store.add_documents(docs)
                    self.loaded_resources.add(pseudo_id)
            except Exception as e:
                logger.error(f"[ResourceLoader] Failed to crawl {url}: {e}")

    def add_pdf(self, file_path: str, original_filename: str):
        """
        Moves a temp file to resource dir and ingests it.
        """
        target_path = PDF_DIR / original_filename
        # Avoid overwrite or handle versioning? For now overwrite.
        shutil.copy(file_path, target_path)
        
        # Trigger load single
        if self.vector_store:
            embedder = self.vector_store.embedder
            docs = process_pdf(str(target_path))
            for d in docs: d.metadata["is_global_resource"] = True
            embeddings = load_embeddings_with_cache(str(target_path), embedder, docs)
            self.vector_store.add_documents_with_embeddings(docs, embeddings)
            embeddings = load_embeddings_with_cache(str(target_path), embedder, docs)
            self.vector_store.add_documents_with_embeddings(docs, embeddings)
            self.loaded_resources.add(original_filename)

    def add_html(self, file_path: str, original_filename: str):
        """
        Moves a temp file to resource html dir and ingests it.
        """
        target_path = HTML_DIR / original_filename
        shutil.copy(file_path, target_path)
        
        # Trigger load single
        if self.vector_store:
            docs = crawl_url(str(target_path))
            for d in docs:
                d.metadata["is_global_resource"] = True
                d.metadata["file_id"] = original_filename
                d.metadata["file_name"] = original_filename
            
            if docs:
                self.vector_store.add_documents(docs)
                self.loaded_resources.add(original_filename)

    def add_url(self, url: str):
        """
        Adds URL to config and ingests.
        """
        config = self._load_config()
        # Check duplicate
        if any(u["url"] == url for u in config["urls"]):
            logger.info(f"URL already exists: {url}")
            # If it exists but not loaded (e.g. restart), we still want to load it?
            # 'load_resources' handles unloaded ones. 
            # If we call add_url, we imply we want to add to config AND load.
            # If already in config, proceed to try loading just in case.
        else:
            config["urls"].append({"url": url, "added_at": str(logging.Formatter().converter())})
            self._save_config(config)
        
        # Trigger load single
        if self.vector_store:
            docs = crawl_url(url)
            pseudo_id = f"url_{hashlib.md5(url.encode('utf-8')).hexdigest()}"
            for d in docs:
                d.metadata["is_global_resource"] = True
                d.metadata["file_id"] = pseudo_id
            
            if docs:
                self.vector_store.add_documents(docs)
                self.loaded_resources.add(pseudo_id)

    def get_resources(self) -> List[Dict]:
        """
        Returns list of available resources.
        """
        res = []
        # PDFs
        for p in PDF_DIR.glob("*.pdf"):
            res.append({"type": "pdf", "name": p.name, "id": p.name})

        # HTMLs
        for h in HTML_DIR.glob("*.html"):
            res.append({"type": "html", "name": h.name, "id": h.name})
        
        # URLs
        config = self._load_config()
        for u in config.get("urls", []):
             res.append({"type": "url", "name": u["url"], "id": f"url_{hashlib.md5(u['url'].encode('utf-8')).hexdigest()}"})
        
        return res

    def delete_resource(self, resource_id: str):
        """
        Deletes a resource (PDF or URL) and removes it from config/disk.
        """
        logger.info(f"[ResourceLoader] Deleting resource: {resource_id}")
        
        # 1. Check if PDF
        pdf_path = PDF_DIR / resource_id
        if pdf_path.exists() and resource_id.endswith(".pdf"):
            try:
                os.remove(pdf_path)
                logger.info(f"[ResourceLoader] Removed PDF file: {resource_id}")
                
                # REMOVE CACHE FILES
                from utils import CACHE_DIR
                cache_file = CACHE_DIR / f"{resource_id}.pkl"
                cache_meta = CACHE_DIR / f"{resource_id}_metadata.pkl"
                # Also embeddings
                emb_cache = CACHE_DIR / f"{resource_id}_embeddings.npy"
                emb_meta = CACHE_DIR / f"{resource_id}_embeddings_meta.json"
                
                for f in [cache_file, cache_meta, emb_cache, emb_meta]:
                    if f.exists():
                        try:
                             os.remove(f)
                             logger.info(f"[ResourceLoader] Removed cache file: {f.name}")
                        except Exception as ce:
                             logger.warning(f"Failed to remove cache {f.name}: {ce}")
                             
                if resource_id in self.loaded_resources:
                    self.loaded_resources.remove(resource_id)
                return True
            except Exception as e:
                logger.error(f"[ResourceLoader] Failed to delete PDF {resource_id}: {e}")
                return False

        # 2. Check if HTML
        html_path = HTML_DIR / resource_id
        if html_path.exists() and (resource_id.endswith(".html") or resource_id.endswith(".htm")):
            try:
                os.remove(html_path)
                logger.info(f"[ResourceLoader] Removed HTML file: {resource_id}")
                if resource_id in self.loaded_resources:
                    self.loaded_resources.remove(resource_id)
                return True
            except Exception as e:
                logger.error(f"[ResourceLoader] Failed to delete HTML {resource_id}: {e}")
                return False

        # 2. Check if URL
        if resource_id.startswith("url_"):
            config = self._load_config()
            original_len = len(config["urls"])
            # Filter out the URL matching the hash
            config["urls"] = [
                u for u in config["urls"] 
                if f"url_{hashlib.md5(u['url'].encode('utf-8')).hexdigest()}" != resource_id
            ]
            
            if len(config["urls"]) < original_len:
                self._save_config(config)
                logger.info(f"[ResourceLoader] Removed URL from config: {resource_id}")
                if resource_id in self.loaded_resources:
                    self.loaded_resources.remove(resource_id)
                return True
        
        logger.warning(f"[ResourceLoader] Resource not found: {resource_id}")
        return False

# Singleton instance?
resource_loader = ResourceLoader()
