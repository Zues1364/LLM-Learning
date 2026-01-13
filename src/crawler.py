import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
import logging
from typing import List
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def crawl_url(url: str) -> List[Document]:
    """
    Fetches content from a URL and parses it into a list of Documents.
    """
    try:
        logger.info(f"Crawling URL: {url}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        # Handle some cases where status code is 403/468 but content might be there? No, usually not.
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "footer", "header"]):
            script.decompose()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading/trailing space on each
        lines = (line.strip() for line in text.splitlines())
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        title = soup.title.string if soup.title else url
        
        # Create a document
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Hash URL to get a stable ID
        url_hash = hashlib.md5(url.encode('utf-8')).hexdigest()
        
        base_doc = Document(
            page_content=text,
            metadata={
                "source": title,
                "url": url,
                "file_id": f"url_{url_hash}", # pseudo file_id
                "type": "web_crawl"
            }
        )
        
        splits = text_splitter.split_documents([base_doc])
        
        # Add index metadata
        for idx, split in enumerate(splits):
             split.metadata["chunk_index"] = idx
             split.metadata["index"] = idx + 1
             
        logger.info(f"Crawled {url}: Generated {len(splits)} chunks.")
        return splits

    except Exception as e:
        logger.error(f"Failed to crawl {url}: {e}")
        # Return empty list on failure, log already handled
        return []
