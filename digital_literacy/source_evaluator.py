import os
import torch
import requests
from bs4 import BeautifulSoup
import logging
from datetime import datetime
import time
from typing import List, Dict, Any
from ddgs import DDGS
import asyncio

from models.model_wrapper import ModelRegistry, LanguageModelWrapper, DEFAULT_GENERATOR, DEFAULT_SENTIMENT
from digital_literacy.browser_controller import BrowserController

log = logging.getLogger(__name__)

# --- ANSI Escape Codes for Colors and Styles ---
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    INVERT = "\033[7m"
    HIDDEN = "\033[8m"
    STRIKETHROUGH = "\033[9m"

    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"

class SourceEvaluator:
    """
    Evaluates academic sources using central memory and models.
    Now equipped with DuckDuckGo for general web searches and 
    Playwright + Qwen VLM for complex site navigation and visual research.
    """

    def __init__(self, device: str = 'cpu'):
        self.device = ModelRegistry.get_device(device)
        log.info(f"SourceEvaluator initialized on device: {self.device}")

        log.info(f"Loading NLP components from Central Registry...")

        # Primary VLM for summarization and logic (Shared with other modules)
        self.vlm = LanguageModelWrapper(model_name=DEFAULT_GENERATOR, device=str(self.device))

        try:
            import os
            import contextlib
            # Silence the "Unexpected weights" BERT load report
            with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
                self.sentiment_model, self.sentiment_processor = ModelRegistry.get_model_and_processor(
                    DEFAULT_SENTIMENT, device=str(self.device), model_type="classification"
                )
            log.info(f"Sentiment/Emotion model (EMOTIONVERSE-2) loaded.")

        except Exception as e:
            log.error(f"Failed to load sentiment model: {e}")
            self.sentiment_model = None

        # DuckDuckGo Search Client
        self.ddgs = DDGS()

    def _fetch_content(self, identifier: str) -> str:
        log.info(f"{Colors.BRIGHT_BLUE}🌐 Attempting to fetch content for: {identifier}{Colors.RESET}")
        if identifier.startswith("http"):
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    # Try with SSL verification first, then fallback to no verification on retry
                    verify = True if attempt == 0 else False
                    response = requests.get(identifier, timeout=10, verify=verify)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
                    text_content = ' '.join([p.get_text() for p in paragraphs])
                    
                    # Basic bot detection check
                    if "please verify you are a human" in text_content.lower() or len(text_content) < 200:
                        raise Exception("Bot detection triggered or insufficient content.")
                    
                    log.info(f"{Colors.GREEN}✅ Content fetched successfully via requests (Attempt {attempt+1}).{Colors.RESET}")
                    return text_content
                except Exception as e:
                    if attempt < max_retries:
                        log.warning(f"{Colors.YELLOW}⚠️  Attempt {attempt+1} failed ({e}). Retrying...{Colors.RESET}")
                        time.sleep(1)
                        continue
                        
                    log.warning(f"{Colors.YELLOW}⚠️  Requests failed ({e}). Falling back to Scaffolding Visual Browser...{Colors.RESET}")
                    # Fallback to Playwright
                    try:
                        # We run the async fetch in a synchronous wrapper
                        return asyncio.run(self._visual_fetch(identifier))
                    except Exception as browser_e:
                        log.error(f"{Colors.RED}❌ Both Requests and Visual Browser failed: {browser_e}{Colors.RESET}")
                        return ""
        else:
            return "Simulated text content regarding advanced AI techniques."

    async def _visual_fetch(self, url: str) -> str:
        """Fallback method that uses Playwright to render the page and extract text."""
        controller = BrowserController(headless=True)
        await controller.start()
        try:
            await controller.navigate(url)
            # Wait for content to render
            await asyncio.sleep(2)
            content = await controller.page.evaluate("document.body.innerText")
            log.info(f"{Colors.GREEN}✅ Content fetched successfully via Visual Browser.{Colors.RESET}")
            return content
        finally:
            await controller.stop()

    def evaluate_source(self, identifier: str, metadata: dict = None) -> dict:
        log.info(f"\n{Colors.BLUE}{Colors.BOLD}--- Initiating Source Evaluation for: {identifier} ---{Colors.RESET}")
        content = self._fetch_content(identifier)
        if not content:
            return {"credibility": "N/A", "bias": "N/A", "relevance": "N/A", "summary": "Could not retrieve content."}

        evaluation_results = {}

        # 1. Credibility using Qwen
        log.info(f"{Colors.CYAN}Assessing credibility via Qwen 3.5...{Colors.RESET}")
        cred_prompt = (
            f"Evaluate the credibility of the following text based on its tone, content, and academic style.\n"
            f"TEXT: {content[:1500]}\n\n"
            f"Classify as: 'highly credible', 'moderately credible', or 'low credibility'.\n"
            f"Credibility:"
        )
        credibility = self.vlm.generate(cred_prompt, max_new_tokens=10, temperature=0.1).strip().lower()
        evaluation_results["credibility"] = credibility

        # 2. Bias Analysis using EMOTIONVERSE + Qwen
        bias_info = "Neutral"
        if self.sentiment_model:
            try:
                inputs = self.sentiment_processor(content[:512], return_tensors="pt", padding=True, truncation=True).to(self.device)
                with torch.no_grad():
                    logits = self.sentiment_model(**inputs).logits
                probs = torch.softmax(logits[0][:8], dim=0)
                labels = ["joy", "trust", "anticipation", "surprise", "anger", "sadness", "fear", "disgust"]
                emotions = {label: prob.item() for label, prob in zip(labels, probs)}
                
                if emotions.get('anger', 0) > 0.3 or emotions.get('fear', 0) > 0.3:
                    bias_info = f"{Colors.RED}Potentially Biased (High emotional charge){Colors.RESET}"
                else:
                    bias_info = f"{Colors.GREEN}Neutral / Objective{Colors.RESET}"
            except:
                bias_info = "Error"
        evaluation_results["bias"] = bias_info

        # 3. Summarization using Qwen 3.5
        log.info(f"{Colors.CYAN}Generating summary via Qwen 3.5 VLM...{Colors.RESET}")
        sum_prompt = (
            f"Please summarize the following text in exactly 3 clear sentences for a research context:\n\n"
            f"TEXT: {content[:2000]}\n\n"
            f"SUMMARY:"
        )
        summary = self.vlm.generate(sum_prompt, max_new_tokens=150, temperature=0.3)
        
        evaluation_results["relevance"] = "High" if len(content) > 1000 else "Medium"
        evaluation_results["summary"] = summary.strip()

        print(f"\n{Colors.GREEN}{Colors.BOLD}--- Source Evaluation Complete ---{Colors.RESET}")
        return evaluation_results

    def search_academic_api(self, query: str, api_type: str = "duckduckgo") -> list:
        log.info(f"\n{Colors.BLUE}{Colors.BOLD}--- Initiating Search for '{query}' on {api_type.upper()} ---{Colors.RESET}")
        results = []
        
        if api_type == "duckduckgo":
            try:
                for r in self.ddgs.text(query, max_results=5):
                    results.append({
                        "title": r.get('title'),
                        "url": r.get('href'),
                        "authors": [r.get('body')[:50] + "..."]
                    })
                log.info(f"{Colors.GREEN}✅ Retrieved {len(results)} results from DuckDuckGo.{Colors.RESET}")
            except Exception as e:
                log.error(f"{Colors.RED}❌ Error searching DuckDuckGo: {e}{Colors.RESET}")

        elif api_type == "arxiv":
            arxiv_url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=5"
            try:
                response = requests.get(arxiv_url, timeout=15)
                soup = BeautifulSoup(response.text, 'xml')
                for entry in soup.find_all('entry'):
                    results.append({
                        "title": entry.title.text.strip(),
                        "url": entry.link['href'],
                        "authors": [a.find('name').text for a in entry.find_all('author')]
                    })
            except Exception as e:
                log.error(f"{Colors.RED}❌ Error searching arXiv: {e}{Colors.RESET}")
                
        return results

    async def visual_browse_and_extract(self, url: str, instruction: str) -> str:
        """
        Uses the visual scaffolding browser to navigate a complex site and return the outcome.
        """
        controller = BrowserController(headless=True)
        await controller.start()
        try:
            await controller.navigate(url)
            log.info(f"Visual Browser asking Qwen what to do about: '{instruction}'")
            action = await controller.decide_and_act(instruction)
            return action
        finally:
            await controller.stop()

if __name__ == "__main__":
    evaluator = SourceEvaluator()
    res = evaluator.search_academic_api("AI in healthcare", "duckduckgo")
    print(res)
