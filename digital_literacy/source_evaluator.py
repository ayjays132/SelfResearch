
import torch
import requests
from bs4 import BeautifulSoup
from transformers import pipeline

class SourceEvaluator:
    """
    A class to evaluate the credibility, bias, and relevance of academic sources.
    Integrates with open academic search APIs and uses NLP for evaluation.
    """

    def __init__(self, device: str = 'cpu'):
        """
        Initializes the SourceEvaluator with a specified device.
        Args:
            device (str): The device to use for computations ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        print(f"SourceEvaluator initialized on device: {self.device}")

        # Initialize NLP pipelines (dummy for now, would be fine-tuned models)
        # Using a generic sentiment analysis as a proxy for bias detection
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=0 if self.device.type == 'cuda' else -1)
        self.summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=0 if self.device.type == 'cuda' else -1)
        self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if self.device.type == 'cuda' else -1)

    def _fetch_content(self, identifier: str) -> str:
        """
        Fetches content from a URL or simulates content for a PDF/metadata.
        Args:
            identifier (str): URL or a dummy identifier for PDF/metadata.
        Returns:
            str: The fetched content.
        """
        if identifier.startswith("http"):
            try:
                response = requests.get(identifier, timeout=10)
                response.raise_for_status() # Raise an exception for HTTP errors
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract text from common tags
                paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
                text_content = ' '.join([p.get_text() for p in paragraphs])
                return text_content
            except requests.exceptions.RequestException as e:
                print(f"Error fetching URL {identifier}: {e}")
                return ""
        else:
            # Simulate content for PDF/metadata - in a real scenario, this would involve PDF parsing or metadata lookup
            return f"Simulated content for {identifier}: This is a research paper discussing advanced topics in AI and machine learning. It presents novel algorithms and experimental results. The authors are from a reputable institution. This paper was published in a peer-reviewed journal and has been cited by many other researchers."

    def evaluate_source(self, identifier: str, metadata: dict = None) -> dict:
        """
        Evaluates a source for credibility, bias, and relevance.
        Args:
            identifier (str): URL, PDF path, or other identifier for the source.
            metadata (dict): Optional metadata about the source (e.g., author, journal).
        Returns:
            dict: A dictionary containing evaluation results.
        """
        content = self._fetch_content(identifier)
        if not content:
            return {"credibility": "N/A", "bias": "N/A", "relevance": "N/A", "summary": "Could not retrieve content."}

        # Simulate NLP analysis with PyTorch tensors
        input_ids = torch.tensor(self.sentiment_analyzer.tokenizer.encode(content, truncation=True, max_length=512)).unsqueeze(0).to(self.device)
        # Dummy computation for demonstration
        dummy_output = torch.sum(input_ids.float() * 0.1)

        # Credibility (using zero-shot classification for more nuanced evaluation)
        credibility_labels = ["highly credible", "moderately credible", "low credibility"]
        credibility_results = self.zero_shot_classifier(content, credibility_labels, multi_label=False)
        credibility = credibility_results["labels"][0] if credibility_results else "N/A"

        # Bias (using sentiment analysis and zero-shot classification for political/ideological bias)
        sentiment_result = self.sentiment_analyzer(content[:512]) # Limit input for sentiment analysis
        bias_sentiment = "Neutral" # Default
        if sentiment_result and sentiment_result[0]["label"] == "NEGATIVE" and sentiment_result[0]["score"] > 0.9:
            bias_sentiment = "Potentially Biased (Negative Tone)"
        elif sentiment_result and sentiment_result[0]["label"] == "POSITIVE" and sentiment_result[0]["score"] > 0.9:
            bias_sentiment = "Potentially Biased (Positive Tone)"

        political_bias_labels = ["left-leaning", "right-leaning", "neutral political bias"]
        political_bias_results = self.zero_shot_classifier(content, political_bias_labels, multi_label=False)
        political_bias = political_bias_results["labels"][0] if political_bias_results else "N/A"

        bias = f"Sentiment: {bias_sentiment}, Political: {political_bias}"

        # Relevance (based on content length and presence of key terms, enhanced with summarization quality)
        relevance_score = len(content) / 1000.0 # Longer content implies more potential relevance
        if metadata and "keywords" in metadata:
            relevance_score += 0.5 # Assume keywords imply relevance
        
        # Check summarization quality as a proxy for coherence and relevance
        summary_result = self.summarizer(content, max_length=100, min_length=30, do_sample=False)
        summary = summary_result[0]["summary_text"] if summary_result else "No summary available."
        if len(summary.split()) > 10: # If summary is substantial, likely more relevant
            relevance_score += 0.3

        relevance = "High" if relevance_score >= 1.5 else ("Medium" if relevance_score >= 0.5 else "Low")

        return {
            "credibility": credibility,
            "bias": bias,
            "relevance": relevance,
            "summary": summary
        }

    def search_academic_api(self, query: str, api_type: str = "arxiv") -> list:
        """
        Searches an academic API for relevant papers.
        Args:
            query (str): The search query.
            api_type (str): The type of API to search ("arxiv" or "semantic_scholar").
        Returns:
            list: A list of search results.
        """
        results = []
        if api_type == "arxiv":
            # Using arXiv API (simplified for demonstration, proper API key and rate limiting would be needed)
            arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
            try:
                response = requests.get(arxiv_url, timeout=10)
                response.raise_for_status()
                # Parse XML response (simplified)
                soup = BeautifulSoup(response.text, 'xml')
                for entry in soup.find_all('entry'):
                    title = entry.title.text.strip()
                    link = entry.link['href']
                    authors = [author.find('name').text for author in entry.find_all('author')]
                    results.append({"title": title, "url": link, "authors": authors})
            except requests.exceptions.RequestException as e:
                print(f"Error searching arXiv: {e}")
            # Simulate some PyTorch computation
            dummy_tensor = torch.randn(5, 5, device=self.device)
            _ = torch.linalg.det(dummy_tensor) # Dummy determinant calculation

        elif api_type == "semantic_scholar":
            # Using Semantic Scholar API (simplified for demonstration, proper API key and rate limiting would be needed)
            semantic_scholar_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=5&fields=title,url,authors"
            try:
                response = requests.get(semantic_scholar_url, timeout=10)
                response.raise_for_status()
                data = response.json()
                for paper in data.get('data', []):
                    title = paper.get('title')
                    url = paper.get('url')
                    authors = [author.get('name') for author in paper.get('authors', [])]
                    results.append({"title": title, "url": url, "authors": authors})
            except requests.exceptions.RequestException as e:
                print(f"Error searching Semantic Scholar: {e}")
            # Simulate some PyTorch computation
            dummy_tensor = torch.randn(7, 7, device=self.device)
            _ = torch.mean(dummy_tensor) # Dummy mean calculation
        return results

if __name__ == "__main__":
    # Example Usage
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA is available! Using GPU.")
    else:
        device = 'cpu'
        print("CUDA not available. Using CPU.")

    evaluator = SourceEvaluator(device=device)

    # Example 1: Evaluate a URL
    print("\n--- Evaluating a URL ---")
    url_to_evaluate = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    evaluation_results = evaluator.evaluate_source(url_to_evaluate)
    print(f"Evaluation for {url_to_evaluate}: {evaluation_results}")

    # Example 2: Evaluate a simulated PDF/metadata
    print("\n--- Evaluating a simulated PDF/metadata ---")
    pdf_identifier = "my_research_paper.pdf"
    pdf_metadata = {"author": "Jane Doe", "journal": "Journal of AI Research", "keywords": ["AI", "ethics", "society"]}
    pdf_evaluation_results = evaluator.evaluate_source(pdf_identifier, metadata=pdf_metadata)
    print(f"Evaluation for {pdf_identifier}: {pdf_evaluation_results}")

    # Example 3: Search academic API
    print("\n--- Searching Academic API ---")
    search_query = "explainable AI"
    arxiv_results = evaluator.search_academic_api(search_query, api_type="arxiv")
    print(f"ArXiv Search Results for '{search_query}': {arxiv_results}")

    semantic_scholar_results = evaluator.search_academic_api(search_query, api_type="semantic_scholar")
    print(f"Semantic Scholar Search Results for '{search_query}': {semantic_scholar_results}")


