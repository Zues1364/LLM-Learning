import requests
import subprocess
from bs4 import BeautifulSoup
from langchain_core.documents import Document
import logging
from typing import List
import hashlib
import os
from bs4.element import Tag, NavigableString

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

    
def _fetch_from_network(url: str):
    """Encapsulates all network fetching attempts (Direct, Curl, Cache)."""
    # List of header profiles to try
    header_profiles = [
        {
            "name": "Googlebot Desktop",
            "User-Agent": "Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",
            "Referer": "https://www.google.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive"
        },
        {
            "name": "Chrome Desktop",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Referer": "https://www.google.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1"
        },
        {
            "name": "Edge Desktop",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edg/120.0.0.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Connection": "keep-alive"
        }
    ]

    response = None
    for headers in header_profiles:
        ua_name = headers.get("name", "Unknown")
        req_headers = {k: v for k, v in headers.items() if k != "name"}
        
        logger.info(f"Crawling URL: {url} using {ua_name}")
        try:
            session = requests.Session()
            resp = session.get(url, headers=req_headers, timeout=15)
            
            if resp.status_code == 200:
                response = resp
                logger.info(f"Success with {ua_name}")
                break
            else:
                logger.warning(f"Failed with {ua_name}: Status {resp.status_code}")
                
        except requests.RequestException as e:
            logger.warning(f"Error with {ua_name}: {e}")
    
    # If direct attempts fail, try Curl Fallback
    if not response or response.status_code != 200:
         logger.warning(f"Requests failed. Attempting Curl fallback for: {url}")
         try:
             cmd = [
                "curl", "-L", "-k",
                "-A", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                url
             ]
             result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
             if result.returncode == 0 and result.stdout.strip():
                 logger.info("Curl fallback successful.")
                 # Mock response object
                 response = type('obj', (object,), {'content': result.stdout.encode('utf-8'), 'status_code': 200, 'text': result.stdout})
             else:
                 logger.warning("Curl fallback failed or returned empty.")
         except Exception as e:
             logger.error(f"Curl fallback exception: {e}")

    # If Curl also fails (or didn't set response), try Google Cache
    if not response or response.status_code != 200:
         logger.warning(f"All direct crawls failed. Attempting Google Cache fallback for: {url}")
         cache_url = f"http://webcache.googleusercontent.com/search?q=cache:{url}&strip=1&vwsrc=0"
         cache_headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"} 
         try:
             response = requests.get(cache_url, headers=cache_headers, timeout=15)
             
             # Handle Redirect
             if response.status_code == 200 and "Google Search" in response.text[:200]:
                 soup_redirect = BeautifulSoup(response.content, 'html.parser')
                 link = soup_redirect.find('a', href=True)
                 if link:
                     redirect_url = link['href']
                     if redirect_url.startswith("/"):
                         redirect_url = "http://webcache.googleusercontent.com" + redirect_url
                     
                     logger.info(f"Following Google Cache redirect to: {redirect_url}")
                     response = requests.get(redirect_url, headers=cache_headers, timeout=15)

             if response.status_code != 200:
                 raise Exception(f"Google Cache failed with status {response.status_code}")
         except Exception as e:
             logger.error(f"Google Cache fallback failed: {e}")
             return None # Ultimately failed

    return response

def _table_to_markdown(table_tag: Tag) -> str:
    """Converts an HTML table to a Markdown pipe table."""
    rows = []
    # Get all rows
    for tr in table_tag.find_all('tr'):
        cells = [cell.get_text(strip=True) for cell in tr.find_all(['th', 'td'])]
        if not cells: continue # Skip empty rows
        
        # Escape pipes in cells to avoid breaking markdown
        cells = [c.replace('|', '&#124;') for c in cells]
        rows.append(f"| {' | '.join(cells)} |")
        
    if not rows: return ""
    
    # Add separator after first row (header assumption)
    # Check column count from first row
    col_count = rows[0].count('|') // 2
    separator = f"| {' | '.join(['---'] * col_count)} |"
    rows.insert(1, separator)
    
    return "\n" + "\n".join(rows) + "\n"

def crawl_url(url: str) -> List[Document]:
    """
    Fetches content from a URL and parses it into a list of Documents.
    """

    
            
    # Check if URL is a local file
    if os.path.isfile(url) or url.startswith("file://"):
        logger.info(f"Crawling local file: {url}")
        file_path = url.replace("file://", "") if url.startswith("file://") else url
        try:
             with open(file_path, 'r', encoding='utf-8') as f:
                 html_content = f.read()
             soup = BeautifulSoup(html_content, 'html.parser')
        except Exception as e:
             logger.error(f"Failed to read local file {url}: {e}")
             return []
    else:
        # Network Crawling
        try:
            response = _fetch_from_network(url)
            if not response:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check for WAF (only relevant for network)
            waf_text = soup.get_text()
            if "EVG WAF" in waf_text or "Truy cập bị chặn" in waf_text or "General Error" in waf_text:
                 logger.warning(f"Detected EVG WAF block page for {url}")
                 raise RuntimeError(f"WAF Blocked: The website '{url}' is blocking our access. Please save the page as HTML (Ctrl+S) and upload it using the 'Upload HTML' button.")
                 
        except Exception as e:
            if "WAF Blocked" in str(e):
                raise e
            logger.error(f"Network crawl failed: {e}")
            return []

    # Common Processing (Cleaning, Tables, Chunking)
        
    # Remove script, style, and navigation elements
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        element.decompose()
            
    # 1. Remove link-heavy lists (often footer links or partner lists)
    for list_tag in soup(["ul", "ol", "div"]):
        # If a list/div has many links relative to text, it's likely a menu or footer
        links = list_tag.find_all("a")
        if len(links) > 5:
            text_len = len(list_tag.get_text())
            # Heuristic: < 20 chars per link on average -> likely a link farm
            if text_len / (len(links) + 1) < 50:
                list_tag.decompose()

    # 2. Process Tables: Convert to Markdown
    for table in soup.find_all('table'):
        markdown_table = _table_to_markdown(table)
        # Replace table with its markdown representation
        # We wrap it in newlines to ensure separation
        new_node = soup.new_string(markdown_table)
        table.replace_with(new_node)
            
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
        
    valid_documents = []
    spam_keywords = ["toto slot", "betting", "casino", "http", "https"]
        
    for idx, doc in enumerate(text_splitter.create_documents([text])):
        # Filter out chunks that are just lists of URLs
        content = doc.page_content.lower()
        lines = content.splitlines()
        url_lines = [l for l in lines if "http" in l]
            
        # If chunk is > 50% URLs, skip it
        if len(url_lines) > len(lines) * 0.5:
            continue
                
        # If chunk contains known spam keywords
        if any(keyword in content for keyword in ["toto slot"]):
            continue

        doc.metadata = {
            "source": title,
            "url": url,
            "file_id": f"url_{url_hash}", # pseudo file_id
            "type": "web_crawl",
            "chunk_index": len(valid_documents),
            "index": len(valid_documents) + 1
        }
        valid_documents.append(doc)

    logger.info(f"Crawled {url}: Generated {len(valid_documents)} valid chunks.")
    return valid_documents


    return valid_documents
