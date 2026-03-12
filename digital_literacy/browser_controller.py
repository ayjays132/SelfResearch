import asyncio
from playwright.async_api import async_playwright
import logging
import base64
import json
from typing import Optional, Dict

from models.model_wrapper import LanguageModelWrapper, DEFAULT_GENERATOR

log = logging.getLogger(__name__)

# CSS and JS to inject a visible cursor and token tags for the VLM
INJECTION_SCRIPT = """
() => {
    // 1. Inject AI Cursor
    if (!document.getElementById('ai-cursor')) {
        const cursor = document.createElement('div');
        cursor.id = 'ai-cursor';
        cursor.style.position = 'fixed';
        cursor.style.top = '50%';
        cursor.style.left = '50%';
        cursor.style.width = '25px';
        cursor.style.height = '25px';
        cursor.style.backgroundColor = 'rgba(255, 0, 0, 0.6)';
        cursor.style.border = '2px solid white';
        cursor.style.borderRadius = '50%';
        cursor.style.zIndex = '999999';
        cursor.style.pointerEvents = 'none';
        cursor.style.transition = 'top 0.3s ease-out, left 0.3s ease-out';
        cursor.style.boxShadow = '0 0 10px rgba(0,0,0,0.5)';
        document.body.appendChild(cursor);
        
        window.updateAiCursor = (x, y) => {
            cursor.style.left = x + 'px';
            cursor.style.top = y + 'px';
        };
    }

    // 2. Inject Bounding Box Token Tags for interactable elements
    let idCounter = 0;
    const interactables = document.querySelectorAll('a, button, input, select, textarea, [role="button"]');
    
    // Remove old tags
    document.querySelectorAll('.ai-token-tag').forEach(e => e.remove());
    
    window.elementMap = {};

    interactables.forEach(el => {
        const rect = el.getBoundingClientRect();
        // Only tag visible elements
        if (rect.width > 0 && rect.height > 0 && rect.top >= 0 && rect.left >= 0) {
            idCounter++;
            const tag = document.createElement('div');
            tag.className = 'ai-token-tag';
            tag.innerText = '[' + idCounter + ']';
            tag.style.position = 'absolute';
            tag.style.top = (rect.top + window.scrollY - 10) + 'px';
            tag.style.left = (rect.left + window.scrollX - 10) + 'px';
            tag.style.backgroundColor = 'yellow';
            tag.style.color = 'black';
            tag.style.fontSize = '12px';
            tag.style.fontWeight = 'bold';
            tag.style.padding = '2px 4px';
            tag.style.borderRadius = '3px';
            tag.style.zIndex = '999998';
            tag.style.pointerEvents = 'none';
            tag.style.border = '1px solid black';
            document.body.appendChild(tag);
            
            // Store mapping of ID to coordinates for Playwright to click
            window.elementMap[idCounter] = {
                x: rect.left + rect.width / 2,
                y: rect.top + rect.height / 2
            };
        }
    });
    return JSON.stringify(window.elementMap);
}
"""

class BrowserController:
    """
    An advanced scaffolding browser controlling system.
    Uses Playwright for interaction and Qwen 3.5 VLM for visual decision making.
    Injects a cursor and numeric tags onto all clickable elements so the VLM 
    can just output 'CLICK [ID]' without guessing pixels.
    """
    def __init__(self, headless: bool = True):
        self.headless = headless
        self.vlm = LanguageModelWrapper(model_name=DEFAULT_GENERATOR)
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.element_map = {}

    async def start(self):
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=self.headless)
        self.context = await self.browser.new_context(viewport={'width': 1280, 'height': 900})
        self.page = await self.context.new_page()
        log.info("Scaffolding Playwright browser started.")

    async def stop(self):
        if self.page: await self.page.close()
        if self.context: await self.context.close()
        if self.browser: await self.browser.close()
        if self.playwright: await self.playwright.stop()
        log.info("Playwright browser stopped.")

    async def navigate(self, url: str):
        log.info(f"Navigating to {url}")
        await self.page.goto(url, wait_until="networkidle")
        await self.refresh_tags()

    async def refresh_tags(self):
        """Re-evaluates the page to draw tags and update the element map."""
        map_json = await self.page.evaluate(INJECTION_SCRIPT)
        self.element_map = json.loads(map_json)

    async def move_cursor(self, x: float, y: float):
        await self.page.mouse.move(x, y)
        await self.page.evaluate(f"window.updateAiCursor({x}, {y})")

    async def click_element(self, element_id: int):
        str_id = str(element_id)
        if str_id in self.element_map:
            coords = self.element_map[str_id]
            await self.move_cursor(coords['x'], coords['y'])
            await self.page.mouse.click(coords['x'], coords['y'])
            log.info(f"Clicked element [{element_id}] at ({coords['x']}, {coords['y']})")
            await asyncio.sleep(2) # wait for page navigation or modal
            await self.refresh_tags()
        else:
            log.warning(f"Element ID [{element_id}] not found on screen.")

    async def type_text(self, text: str):
        await self.page.keyboard.type(text)
        log.info(f"Typed text: {text}")

    async def get_screenshot_b64(self) -> str:
        img_bytes = await self.page.screenshot(type="jpeg", quality=85)
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{b64_img}"

    async def decide_and_act(self, task_instruction: str) -> str:
        """
        Takes a tagged screenshot, asks the VLM what to do, and executes the action.
        Uses element IDs for flawless clicking.
        """
        screenshot_url = await self.get_screenshot_b64()
        
        prompt = (
            f"You are an AI research browser agent. Task: '{task_instruction}'.\n"
            f"The image shows a webpage where interactable elements have yellow tags with numbers (e.g. [1]).\n"
            f"Decide your next action based on the screenshot.\n"
            f"Respond STRICTLY in one of the following formats:\n"
            f"- CLICK <number> (e.g., CLICK 5)\n"
            f"- TYPE <text> (e.g., TYPE AI Research)\n"
            f"- SCROLL_DOWN\n"
            f"- DONE\n"
            f"Action:"
        )

        response = self.vlm.generate(prompt=prompt, image_url=screenshot_url, max_new_tokens=20)
        action = response.strip().upper()
        log.info(f"VLM decided action: {action}")

        if action.startswith("CLICK"):
            parts = action.split()
            if len(parts) >= 2:
                try:
                    el_id = int(parts[1].strip("[]"))
                    await self.click_element(el_id)
                except ValueError:
                    log.error("Failed to parse element ID.")
        elif action.startswith("TYPE"):
            text = response[5:].strip().strip("'\"") # use original casing for typing
            await self.type_text(text)
            # Automatically hit Enter after typing in many cases
            await self.page.keyboard.press("Enter")
            await asyncio.sleep(2)
            await self.refresh_tags()
        elif action.startswith("SCROLL_DOWN"):
            await self.page.mouse.wheel(0, 700)
            await asyncio.sleep(1)
            await self.refresh_tags()
        elif action == "DONE":
            return "Task completed."
        
        return action

    async def extract_accessibility_tree(self) -> str:
        """Extracts a semantic, text-based representation of the DOM."""
        if not self.page: return "Error: Browser not started."
        
        # We inject a script to walk the DOM and build a semantic tree
        # This bypasses CSS obfuscation and React virtual DOMs
        script = """
        () => {
            function buildTree(node, depth) {
                if (depth > 8) return ''; // limit depth to prevent massive output
                let result = '';
                const role = node.getAttribute ? node.getAttribute('role') : null;
                const tag = node.tagName ? node.tagName.toLowerCase() : '';
                
                // Focus on semantic, readable nodes
                if (['p', 'h1', 'h2', 'h3', 'h4', 'a', 'button', 'li', 'article', 'section', 'main'].includes(tag) || role) {
                    let text = node.innerText || node.textContent || '';
                    // Clean up text
                    text = text.replace(/\\s+/g, ' ').trim();
                    if (text.length > 0 && text.length < 500) { // skip massive raw text blocks at parent levels
                        let indent = '  '.repeat(depth);
                        let descriptor = role ? `[role="${role}"]` : `<${tag}>`;
                        
                        // Grab our custom injected AI ID if it exists
                        const aiId = node.getAttribute('ai-token-id');
                        if (aiId) descriptor += ` [ID:${aiId}]`;
                        
                        result += `${indent}${descriptor} ${text.substring(0, 100)}\\n`;
                    }
                }
                
                // Only traverse semantic children to avoid massive div soup
                if (['div', 'span', 'article', 'section', 'main', 'ul', 'ol', 'nav', 'header', 'footer'].includes(tag) || tag === 'body') {
                    for (let child of node.children) {
                        result += buildTree(child, depth + 1);
                    }
                }
                return result;
            }
            return buildTree(document.body, 0);
        }
        """
        try:
            tree = await self.page.evaluate(script)
            return tree[:3000] + "\n...[truncated]" if len(tree) > 3000 else tree
        except Exception as e:
            return f"Failed to extract accessibility tree: {e}"

    async def autonomous_search_loop(self, query: str, max_steps: int = 5) -> str:
        """
        Runs an autonomous visual loop:
        Navigates to a search engine, types the query, and browses.
        """
        import rich
        from rich.panel import Panel
        from rich.console import Console
        c = Console()
        c.print(Panel(f"[bold yellow]Initializing Autonomous Visual Browser Loop[/bold yellow]\nGoal: {query}", border_style="yellow"))
        
        await self.start()
        await self.navigate("https://duckduckgo.com")
        
        extracted_data = ""
        
        for step in range(max_steps):
            c.print(f"[dim]Visual Loop Step {step+1}/{max_steps}...[/dim]")
            
            # Fetch semantic context
            a11y_tree = await self.extract_accessibility_tree()
            
            prompt = (
                f"Goal: Find research regarding: {query}.\n"
                f"Semantic Page Structure:\n{a11y_tree}\n\n"
                "If on search engine, click search box (look for input/button tags), type query. "
                "If results shown, click a relevant link by ID. "
                "If you have enough info, output DONE."
            )
            
            action = await self.decide_and_act(prompt)
            c.print(f"[bold cyan]Agent Action:[/bold cyan] {action}")
            if action == "Task completed." or action.upper() == "DONE":
                break
        
        # Final extraction
        content = await self.page.evaluate("document.body.innerText")
        extracted_data = content[:4000] + "\n...[truncated]"
        
        await self.stop()
        
        c.print(Panel(f"[bold green]Visual Browsing Complete.[/bold green]\nExtracted context length: {len(content)}", border_style="green"))
        return extracted_data

