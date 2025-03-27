import requests
import urllib.parse

API_KEY = "b91e335ef3ef0b0f01dceef77c1c057d0d538bed"  # Lưu ý: sử dụng biến môi trường cho sản xuất

def web_search(query, num_results=3):
    encoded_query = urllib.parse.quote(query)
    url = f"https://google.serper.dev/search?q={encoded_query}&apiKey={API_KEY}"
    response = requests.get(url)
    try:
        json_data = response.json()
        results = json_data.get("organic", [])
        snippets = [item.get("snippet", "") for item in results[:num_results]]
        return snippets
    except Exception as e:
        print("Error parsing web search result:", e)
        return ["(No web results)"]

# Test hàm web_search
if __name__ == "__main__":
    query = "tell me about dog"
    snippets = web_search(query, num_results=3)
    print("Web Search Results:")
    for snippet in snippets:
        print("-", snippet)
