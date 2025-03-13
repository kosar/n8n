#!/usr/bin/env python3

import re
import os
import sys
import json
import time
import psutil
import asyncio
import aiohttp
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import pytesseract
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.text import Text  # Add this import

# Define DisplayManager class here
class DisplayManager:
    """Unified display manager for consistent UI across all application states"""
    def __init__(self):
        self.layout = Layout()
        self.layout.split(
            Layout(name="progress", ratio=1),
            Layout(name="main", ratio=2)  # Increased ratio for more debug space
        )
        
        # Progress tracking components
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="cyan", finished_style="green"),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
        )
        
        self.detail_progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="yellow", finished_style="green"),
            TimeElapsedColumn(),
        )
        
        # Setup progress group
        progress_group = Table.grid()
        progress_group.add_row(Panel(self.progress, title="Overall Progress"))
        progress_group.add_row(Panel(self.detail_progress, title="Current Task"))
        
        # Keep status manager for future extension possibilities
        self.status_manager = StatusManager()
        
        # Debug message storage
        self.debug_messages = []
        
        # Configure the layout
        self.layout["progress"].update(progress_group)
        self.layout["main"].update(Panel(Text(), title="Debug Output"))
        
        # Live display
        self.live = None
        
    def start(self, refresh_per_second=4):
        """Start the live display"""
        self.live = Live(self.layout, refresh_per_second=refresh_per_second)
        self.live.start()
        return self.live
        
    def stop(self):
        """Stop the live display"""
        if self.live:
            self.live.stop()
            
    async def start_display_updater(self):
        """Start the display updater task - no longer needed but kept for API compatibility"""
        # Just return a dummy completed task
        return asyncio.create_task(asyncio.sleep(0))
            
    def add_debug(self, msg: str):
        """Add debug message"""
        if not self.live:
            console.print(msg)
            return
            
        plain_msg = Text.from_markup(msg).plain
        timestamp = datetime.now().strftime('%H:%M:%S')
        self.debug_messages.append(f"{timestamp} {plain_msg}")
        self.layout["main"].update(Panel(Text("\n".join(self.debug_messages[-10:])), title="Debug Output"))
        if self.live:
            self.live.refresh()
            
    async def update_status(self, component: str, status: str):
        """Log status changes to debug output instead of using a separate status display"""
        self.add_debug(f"[dim]{component}: {status}[/dim]")

# Constants
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.JPG', '.PNG', '.JPEG', '.TIFF', '.pdf', '.PDF'}
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/ocr_results")
CONFIG_DIR = os.path.expanduser("~/.ocr_extractor")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")
DEFAULT_MODEL = "mistral-small" # A single place to define the default model

# Enhanced prompt for chronological conversation reconstruction
ANALYSIS_PROMPT = """
You are an expert forensic text analyst specializing in reconstructing conversations from fragmentary evidence.

I have OCR results from multiple text message screenshots that may be incomplete, cut off, or in different formats.

Your task:
1. Create a strict chronological timeline of the conversation
2. Identify all participants by carefully analyzing message patterns, headers, and content
3. For each message, determine the sender and recipients based on context clues
4. Connect related messages into coherent conversation threads
5. Use content analysis to place ambiguous messages in the correct sequence
6. Fill in reasonable gaps in the conversation using context

Format your analysis as a beautiful, narrative-style markdown document:
- Begin with an executive summary of the conversation with the full timeline as you see it
- Create distinct sections for each conversation thread or topic to maintain clarity
- Use clear headings with timestamps
- Visually distinguish different speakers
- Include quoted text blocks for important messages
- Use markdown tables where appropriate to organize information
- Use bold and italics to highlight key points and decisions
- Refer to the source images or OCR data as needed for context so users can click through. Use links.

This task requires a keen eye for detail, logical reasoning, and a touch of creativity to fill in the gaps. Good luck!

Make sure the markdown is well-formatted, easy to read, and captures the essence of the conversation. It should be absolutely gorgeous too. 
"""

# Initialize Rich console
console = Console()

def load_config() -> Dict[str, Any]:
    """Load configuration from file"""
    default_config = {
        "last_directory": os.getcwd(),
        "output_directory": DEFAULT_OUTPUT_DIR,
        "last_language": "eng",
        "selected_model": DEFAULT_MODEL
    }
    
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            save_config(default_config)
            return default_config
    except Exception as e:
        console.print(f"[yellow]Warning: Could not load config: {str(e)}[/yellow]")
        return default_config

def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file"""
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not save config: {str(e)}[/yellow]")

def banner():
    """Display the application banner"""
    return Panel.fit(
        "[bold blue]OCR Text Extractor[/bold blue] - Extract text from images to reconstruct message timelines",
        border_style="green"
    )

def is_supported_file(file_path: str) -> bool:
    """Check if a file is a supported image type"""
    return os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in map(str.lower, SUPPORTED_EXTENSIONS)

def check_tesseract_installation() -> bool:
    """Check if Tesseract OCR is properly installed"""
    try:
        version = pytesseract.get_tesseract_version()
        console.print(f"[green]✓ Tesseract OCR detected (version {version})[/green]")
        return True
    except Exception as e:
        console.print("[bold red]Tesseract OCR not found or not properly configured.[/bold red]")
        console.print("[yellow]Please install Tesseract OCR and ensure it's in your PATH.[/yellow]")
        console.print("[blue]macOS installation:[/blue]")
        console.print("brew install tesseract")
        return False
        
def check_ollama_status() -> bool:
    """Check if Ollama server is running and display model information"""
    try:
        import ollama
        config = load_config()
        selected_model = config.get("selected_model", DEFAULT_MODEL)
        
        models = get_available_models()
        if models:
            model_count = len(models)
            model_list = ", ".join(models[:3]) + ("..." if len(models) > 3 else "")
            console.print(f"[green]✓ Ollama server detected ({model_count} models available: {model_list})[/green]")
            console.print(f"[green]  Current model: {selected_model}[/green]")
            return True
        else:
            console.print("[yellow]⚠ Ollama server found but no models available.[/yellow]")
            return False
    except ImportError:
        console.print("[yellow]⚠ Ollama Python package not installed.[/yellow]")
        console.print("[yellow]  Install with: pip install ollama[/yellow]")
        return False
    except Exception as e:
        console.print(f"[yellow]⚠ Ollama server not accessible: {str(e)}[/yellow]")
        return False

def get_current_model() -> str:
    """Get the current model from config or return default if unavailable"""
    try:
        config = load_config()
        return config.get("selected_model", DEFAULT_MODEL)
    except Exception:
        return DEFAULT_MODEL

def select_directory() -> str:
    """Simple directory selection via prompt"""
    console.print("[yellow]Enter directory path:[/yellow]")
    directory = input("> ")
    directory = os.path.expanduser(directory)
    
    if not os.path.isdir(directory):
        console.print("[red]Invalid directory.[/red]")
        return os.getcwd()
    
    file_count = len(get_all_images_in_directory(directory))
    console.print(f"[green]Found {file_count} supported image files in the selected directory.[/green]")
    
    return directory

def get_all_images_in_directory(directory: str) -> List[str]:
    """Get all supported image files in a directory"""
    return [os.path.join(directory, f) for f in os.listdir(directory) 
            if is_supported_file(os.path.join(directory, f))]

def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess the image to improve OCR results"""
    image = cv2.imread(image_path)
    if image is None:
        try:
            pil_image = Image.open(image_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            if image is None:
                raise ValueError(f"Could not open image: {image_path}")
        except Exception as e:
            raise ValueError(f"Could not open image: {image_path} - {e}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return adaptive_thresh

async def extract_text_from_image(image_path: str, lang: str = "eng", progress: Optional[Progress] = None, task_id: Optional[int] = None) -> str:
    """Extract text from image using OCR"""
    try:
        if progress:
            progress.update(task_id, description=f"[blue]Preprocessing image...")
            
        if image_path.lower().endswith('.pdf'):
            from pdf2image import convert_from_path
            if progress:
                progress.update(task_id, description=f"[blue]Converting PDF to images...")
            pages = convert_from_path(image_path, 300)
            if progress:
                progress.update(task_id, description=f"[blue]Performing OCR on {len(pages)} PDF pages...")
            text_results = [pytesseract.image_to_string(page, lang=lang) for page in pages]
            combined_text = "\n\n".join(text_results)
        else:
            if progress:
                progress.update(task_id, description=f"[blue]Preprocessing image...")
            preprocessed_image = preprocess_image(image_path)
            
            # Use pytesseract
            if progress:
                progress.update(task_id, description=f"[blue]Running Tesseract OCR...")
            configs = ['--psm 6 --oem 3', '--psm 4 --oem 3', '--psm 3 --oem 3']
            tesseract_texts = [pytesseract.image_to_string(preprocessed_image, lang=lang, config=cfg) for cfg in configs]
            tesseract_text = max(tesseract_texts, key=len)
            
            # Combine the results
            combined_text = tesseract_text

        # Post-process and clean with Ollama
        if progress:
            progress.update(task_id, description=f"[blue]Cleaning text with Ollama...")
        cleaned_text = await clean_text_with_ollama(combined_text, progress, task_id)

        if progress:
            progress.update(task_id, description=f"[blue]Post-processing text...")
        return post_process_ocr_text(cleaned_text)
    except Exception as e:
        console.print(f"[bold red]Error processing {image_path}: {str(e)}[/bold red]")
        return f"ERROR: {str(e)}"

async def clean_text_with_ollama(
    text: str,
    progress: Optional[Progress] = None,
    task_id: Optional[int] = None,
    display_manager: Optional[DisplayManager] = None  # <-- Added parameter for debug output
) -> str:
    """Clean the OCR text using Ollama with better error reporting"""
    try:
        import ollama
    except ImportError:
        console.print("[yellow]Ollama not installed. Install with: pip install ollama[/yellow]")
        return text
        
    try:
        model_name = get_current_model()
        
        if progress:
            progress.update(task_id, description=f"[magenta]Ollama: Preparing to clean text...")
        if display_manager:
            display_manager.add_debug("[blue]Ollama OCR cleanup: Initiating cleanup process...[/blue]")
        
        prompt = f"You are a helpful assistant. Clean up this OCR text:\n\n{text}\n\nCleaned text:"
        
        try:
            start_time = time.time()
            cleaned_text = await asyncio.wait_for(
                stream_ollama_completion(
                    model_name, 
                    prompt, 
                    temperature=0.1,
                    progress=progress,
                    task_id=task_id
                ),
                timeout=60  # 60 second timeout
            )
            if display_manager:
                elapsed = time.time() - start_time
                display_manager.add_debug(f"[blue]Ollama OCR cleanup: Completed in {elapsed:.2f}s[/blue]")
            return cleaned_text
            
        except asyncio.TimeoutError:
            if progress:
                progress.update(task_id, description=f"[red]Ollama: Timed out after 60s, using original text")
            if display_manager:
                display_manager.add_debug("[red]Ollama OCR cleanup: Timed out after 60s[/red]")
            return text
            
        except Exception as e:
            if progress:
                progress.update(task_id, description=f"[red]Ollama error: {str(e)}, using original text")
            if display_manager:
                display_manager.add_debug(f"[red]Ollama OCR cleanup: Error - {str(e)}[/red]")
            return text
            
    except Exception as e:
        if progress:
            progress.update(task_id, description=f"[red]Failed to clean text: {str(e)}")
        return text

async def stream_ollama_completion(model: str, prompt: str, temperature: float = 0.2, progress: Optional[Progress] = None, task_id: Optional[int] = None) -> str:
    """Stream a completion from Ollama with improved error handling"""
    result = ""
    tokens = 0
    start_time = time.time()
    last_update = start_time
    
    try:
        async with aiohttp.ClientSession() as session:
            if progress:
                progress.update(task_id, description=f"[cyan]Connecting to Ollama...")
                
            async with session.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': model,
                    'prompt': prompt,
                    'stream': True,
                    'options': {
                        'temperature': temperature
                    }
                },
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Ollama API error {response.status}: {error_text}")
                
                if progress:
                    progress.update(task_id, description=f"[cyan]Starting token generation...")
                
                async for line in response.content:
                    if line:
                        try:
                            data = json.loads(line)
                            
                            if 'response' in data:
                                result += data['response']
                                tokens += 1
                                
                                # Update progress every second
                                current_time = time.time()
                                if current_time - last_update >= 1.0:
                                    if progress and task_id:
                                        elapsed = current_time - start_time
                                        rate = tokens / elapsed if elapsed > 0 else 0
                                        progress.update(
                                            task_id, 
                                            description=f"[cyan]Tokens: {tokens} ({rate:.1f}/sec)"
                                        )
                                    last_update = current_time
                            
                            if data.get('done', False):
                                if progress and task_id:
                                    elapsed = time.time() - start_time
                                    rate = tokens / elapsed if elapsed > 0 else 0
                                    progress.update(
                                        task_id,
                                        description=f"[green]Complete: {tokens} tokens in {elapsed:.1f}s ({rate:.1f}/sec)"
                                    )
                                break
                                
                        except json.JSONDecodeError as e:
                            if progress:
                                progress.update(task_id, description=f"[yellow]Warning: Invalid JSON received")
                            
    except aiohttp.ClientConnectorError:
        raise ConnectionError("Could not connect to Ollama server")
    except Exception as e:
        raise RuntimeError(f"Ollama streaming error: {str(e)}")
    
    return result

class StatusUpdate:
    """Immutable status update message"""
    def __init__(self, component: str, status: str):
        self._component = component
        self._status = status
        self._timestamp = datetime.now()
    
    @property
    def as_row(self) -> Tuple[str, str]:
        return (self._component, self._status)

class StatusManager:
    """Thread-safe status manager using message queue"""
    def __init__(self):
        self._updates = asyncio.Queue()
        self._current_status = []
        self._lock = asyncio.Lock()
    
    async def update(self, component: str, status: str):
        await self._updates.put(StatusUpdate(component, status))
    
    async def get_status_rows(self) -> List[Tuple[str, str]]:
        async with self._lock:
            while not self._updates.empty():
                update = await self._updates.get()
                # Replace existing status for component or add new
                self._current_status = [s for s in self._current_status if s._component != update._component]
                self._current_status.append(update)
            return [s.as_row for s in self._current_status]

async def update_display(layout: Layout, status_manager: StatusManager, live: Live):
    """Legacy function - kept for compatibility"""
    await asyncio.sleep(0.1)  # Do nothing

async def process_images(file_paths: List[str], lang: str = "eng",
                         display_manager: Optional[DisplayManager] = None) -> List[Dict[str, Any]]:
    """Process images with thread-safe status updates"""
    if not file_paths:
        console.print("[yellow]No files selected for processing.[/yellow]")
        return []
    
    results = []
    
    # Create display manager if not provided
    using_local_display = display_manager is None
    if using_local_display:
        display_manager = DisplayManager()
        
    try:
        if using_local_display:
            live = display_manager.start()
            display_task = await display_manager.start_display_updater()
            
        # Initial status
        await display_manager.update_status("Status", "Starting processing...")
        
        # Process files
        main_task = display_manager.progress.add_task(
            f"[cyan]Processing {len(file_paths)} images...", 
            total=len(file_paths)
        )
        
        display_manager.add_debug("[cyan]Starting processing...[/cyan]")
        
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            display_manager.add_debug(f"[blue]Processing {filename}...[/blue]")
            
            # Add subtasks for better tracking
            ocr_task = display_manager.detail_progress.add_task(
                f"[blue]OCR: {filename}", 
                total=1,
                start=False
            )
            
            try:
                display_manager.detail_progress.start_task(ocr_task)
                ocr_text = await extract_text_from_image(
                    file_path, lang, display_manager.detail_progress, ocr_task
                )
                
                if "ERROR:" in ocr_text:
                    display_manager.add_debug(f"[red]OCR failed for {filename}: {ocr_text}[/red]")
                else:
                    display_manager.add_debug(f"[green]OCR completed for {filename}[/green]")
                
                display_manager.detail_progress.update(ocr_task, completed=1)
                
                metadata = get_image_metadata(file_path)
                result = {
                    "filename": filename,
                    "file_path": file_path,
                    "metadata": metadata,
                    "text": ocr_text,
                    "processed_at": datetime.now().isoformat()
                }
                results.append(result)
                
                # Update status
                await display_manager.update_status("Processing", f"Processing {filename}")
                await display_manager.update_status("Progress", f"Completed {len(results)}/{len(file_paths)}")
                
            except Exception as e:
                error_msg = f"Error processing {filename}: {str(e)}"
                display_manager.add_debug(f"[red]{error_msg}[/red]")
                results.append({
                    "filename": filename,
                    "file_path": file_path,
                    "error": str(e),
                    "processed_at": datetime.now().isoformat()
                })
                await display_manager.update_status("Error", f"Failed: {str(e)[:30]}")
            
            display_manager.progress.advance(main_task)
        
        display_manager.add_debug("[green]Processing complete![/green]")
        
        # Clean up if we created the display
        if using_local_display:
            # Clean up display task
            display_task.cancel()
            try:
                await asyncio.wait_for(display_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            display_manager.stop()
            
    except Exception as e:
        console.print(f"[bold red]Display error: {str(e)}[/bold red]")
        console.print(traceback.format_exc())
        if using_local_display:
            display_manager.stop()
    
    return results

async def monitor_ollama_status(task_id, progress, stop_event, display_manager=None):
    """Enhanced monitoring for Ollama server with detailed system performance indicators"""
    try:
        import psutil
        import aiohttp
        
        last_stats = {}
        last_update_time = time.time()
        start_time = time.time()
        previous_tokens = 0
        
        # First, immediately clear the "Connecting..." message
        if progress:
            progress.update(task_id, description=f"[green]Ollama connected")
        
        # Also log initial connection success to debug
        if display_manager:
            display_manager.add_debug(f"[green]✓ Connected to Ollama server[/green]")
        
        status_update_counter = 0
        
        while not stop_event.is_set():
            metrics = {}
            current_time = time.time()
            elapsed = current_time - start_time
            status_update_counter += 1
            
            # Track the model being used
            metrics["model"] = get_current_model()
            
            # Get process info - improved process detection
            try:
                found = False
                for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
                    proc_info = proc.info
                    if 'name' in proc_info and proc_info['name'] and 'ollama' in proc_info['name'].lower():
                        found = True
                        # Update CPU usage before reading
                        proc.cpu_percent()
                        # Wait a moment for accuracy
                        await asyncio.sleep(0.5)
                        cpu_percent = proc.cpu_percent()
                        metrics["cpu"] = cpu_percent
                        
                        if 'memory_info' in proc_info and proc_info['memory_info']:
                            mem_mb = proc_info['memory_info'].rss / (1024 * 1024)
                            metrics["memory"] = round(mem_mb)
                        break
                
                if not found:
                    metrics["status"] = "Process not found"
            except Exception as e:
                metrics["error"] = str(e)[:20]
                
            # Try API endpoints for metrics
            try:
                async with aiohttp.ClientSession() as session:
                    # Check runtime metrics from Prometheus endpoint
                    async with session.get('http://localhost:11434/api/metrics', timeout=2) as response:
                        if response.status == 200:
                            text = await response.text()
                            
                            # Parse Prometheus format metrics
                            total_tokens = 0
                            
                            for line in text.split('\n'):
                                if not line or line.startswith('#'):
                                    continue
                                    
                                # Extract key metrics
                                if "ollama_tokens_total" in line and "type=\"completion\"" in line:
                                    try:
                                        total_tokens = int(float(line.split('}')[1].strip()))
                                        metrics["tokens"] = total_tokens
                                        
                                        # Calculate token rate
                                        if elapsed > 0 and total_tokens > 0:
                                            # Overall rate
                                            overall_rate = total_tokens / elapsed
                                            metrics["overall_rate"] = overall_rate
                                            
                                            # Recent rate (since last check)
                                            if previous_tokens > 0:
                                                tokens_delta = total_tokens - previous_tokens
                                                time_delta = current_time - last_update_time
                                                if time_delta > 0:
                                                    recent_rate = tokens_delta / time_delta
                                                    metrics["token_rate"] = recent_rate
                                            previous_tokens = total_tokens
                                    except Exception:
                                        pass
                                        
                                elif "ollama_gpu_usage" in line:
                                    try:
                                        metrics["gpu"] = round(float(line.split('}')[1].strip()), 1)
                                    except Exception:
                                        pass
            except Exception:
                pass
                
            # Format metrics for progress bar display
            status_parts = []
            if "model" in metrics:
                status_parts.append(f"{metrics['model']}")
            if "token_rate" in metrics:
                status_parts.append(f"{metrics['token_rate']:.1f}t/s")
            if "tokens" in metrics:
                status_parts.append(f"{metrics['tokens']} tokens")
            if "gpu" in metrics:
                status_parts.append(f"GPU:{metrics['gpu']}%")
            if "cpu" in metrics:
                status_parts.append(f"CPU:{metrics['cpu']:.1f}%")
            if "memory" in metrics:
                status_parts.append(f"Mem:{metrics['memory']}MB")
            
            # Always update progress display
            if not status_parts:
                status_parts.append("Monitoring...")
            status_text = " | ".join(status_parts)
            progress.update(task_id, description=f"[cyan]Ollama: {status_text}")
            
            # Log to debug output periodically rather than using a separate panel
            # Every 5th update (about every ~10 seconds) or when significant changes occur
            should_log = (status_update_counter % 5 == 0)
            
            # Also log if there are significant changes in performance
            if ("token_rate" in metrics and "last_rate" in last_stats and 
                abs(metrics["token_rate"] - last_stats["last_rate"]) > 2.0):
                should_log = True
            
            if display_manager and should_log:
                log_parts = []
                
                # Format a comprehensive but concise status line
                if "model" in metrics:
                    log_parts.append(f"[cyan]{metrics['model']}[/cyan]")
                
                if "tokens" in metrics:
                    log_parts.append(f"{metrics['tokens']} tokens")
                    
                if "token_rate" in metrics:
                    rate_info = f"{metrics['token_rate']:.1f} tokens/sec"
                    # Add trend indicators
                    if "last_rate" in last_stats:
                        change = metrics["token_rate"] - last_stats["last_rate"] 
                        if abs(change) > 0.5:
                            direction = "↑" if change > 0 else "↓"
                            color = "green" if change > 0 else "red"
                            rate_info += f" [{color}]{direction}{abs(change):.1f}[/{color}]"
                    log_parts.append(rate_info)
                
                hardware_parts = []
                if "gpu" in metrics:
                    hardware_parts.append(f"GPU: [magenta]{metrics['gpu']}%[/magenta]")
                if "cpu" in metrics:
                    hardware_parts.append(f"CPU: {metrics['cpu']:.1f}%")
                if "memory" in metrics:
                    hardware_parts.append(f"Mem: {metrics['memory']} MB")
                
                if hardware_parts:
                    log_parts.append("(" + ", ".join(hardware_parts) + ")")
                
                # Add it to the debug output
                display_manager.add_debug(f"[dim]Ollama status: {' | '.join(log_parts)}[/dim]")
            
            # Save for next comparison
            if "token_rate" in metrics:
                last_stats["last_rate"] = metrics["token_rate"]
            
            last_update_time = current_time
            
            # Check stop event with a timeout
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=1.5)  # Check more frequently
            except asyncio.TimeoutError:
                pass
            
    except Exception as e:
        if progress:
            progress.update(task_id, description=f"[red]Monitor error: {str(e)}")
        if display_manager:
            display_manager.add_debug(f"[yellow]Ollama monitor error: {str(e)}[/yellow]")

async def analyze_with_ollama(results_file: str, output_dir: str, 
                              display_manager: Optional[DisplayManager] = None):
    """Analyze OCR results with Ollama - unified display approach"""
    try:
        import ollama
        import aiohttp
        import json
        
        using_local_display = display_manager is None
        if using_local_display:
            display_manager = DisplayManager() 
            live = display_manager.start()
            display_task = await display_manager.start_display_updater()
            
        await display_manager.update_status("Status", "Starting analysis...")
        display_manager.add_debug("[cyan]Connecting to Ollama server...[/cyan]")
        
        # Load results
        load_task = display_manager.progress.add_task("[cyan]Loading OCR results...", total=1)
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results = data.get("results", [])
        display_manager.progress.update(load_task, completed=1)

        file_paths = [result.get("file_path") for result in results if "file_path" in result]
        
        # Step 1: Create summary and copy images
        copy_task = display_manager.progress.add_task("[cyan]Creating output resources...", total=2)
        
        display_manager.add_debug("[cyan]Copying images to output directory...[/cyan]")
        display_manager.progress.update(copy_task, description="[cyan]Copying images to output directory...")
        image_map = copy_images_to_output(file_paths, output_dir)
        display_manager.progress.update(copy_task, advance=1)

        display_manager.add_debug("[cyan]Creating markdown summary...[/cyan]")
        display_manager.progress.update(copy_task, description="[cyan]Creating markdown summary...")
        markdown_path = await create_markdown_summary(results, results_file, image_map)
        display_manager.progress.update(copy_task, advance=1, description="[green]Output resources prepared")
        display_manager.add_debug(f"[green]✓ Created initial markdown summary: {markdown_path}[/green]")

        # Step 2: Prepare the prompt
        prompt_task = display_manager.progress.add_task("[cyan]Building analysis prompt...", total=1)
        
        # Read postprompt.txt if it exists
        post_prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "postprompt.txt")
        post_prompt = ""
        has_custom_prompt = False
        if (os.path.exists(post_prompt_file)):
            try:
                with open(post_prompt_file, 'r', encoding='utf-8') as f:
                    post_prompt = f.read()
                if post_prompt.strip():
                    has_custom_prompt = True
                    display_manager.add_debug(f"[green]✓ Loaded custom prompt from {post_prompt_file}[/green]")
            except Exception as e:
                display_manager.add_debug(f"[yellow]Error reading postprompt.txt: {str(e)}[/yellow]")
        else:
            display_manager.add_debug(f"[blue]Using default prompt only (no postprompt.txt found)[/blue]")

        # Generate the prompt with simplified data
        simplified_data = []
        for result in results:
            simplified_entry = {
                "filename": result.get("filename", "unknown"),
                "text": result.get("text", ""),
            }
            if "date" in result:
                simplified_entry["date"] = result["date"]
            if "sender" in result:
                simplified_entry["sender"] = result["sender"]
            simplified_data.append(simplified_entry)

        prompt = ANALYSIS_PROMPT + "\n\nHere's the OCR data extracted from screenshots:\n"
        prompt += json.dumps(simplified_data, indent=2)
        
        if post_prompt:
            prompt += "\n\n" + post_prompt
        
        # Display prompt statistics
        prompt_size = len(prompt)
        token_estimate = prompt_size // 4  # rough estimate
        display_manager.progress.update(prompt_task, completed=1, description=f"[green]Prompt ready: {prompt_size/1000:.1f}K chars (~{token_estimate} tokens)")
        
        # Step 3: Set up the model
        model_name = get_current_model()
        model_task = display_manager.progress.add_task(f"[cyan]Setting up model: {model_name}...", total=1)

        try:
            models = ollama.list()
            model_names = [m.model for m in models if hasattr(m, 'model')]

            if model_name not in model_names:
                display_manager.progress.update(model_task, description=f"[yellow]Pulling model: {model_name}...")
                ollama.pull(model_name)
            display_manager.progress.update(model_task, completed=1, description=f"[green]Model {model_name} ready")
        except Exception as e:
            display_manager.add_debug(f"[yellow]Error checking models, will try to use {model_name} directly: {str(e)}[/yellow]")
            display_manager.progress.update(model_task, completed=1, description=f"[yellow]Model status unknown")
        
        # Step 4: Run the analysis with detailed monitoring
        analysis_task = display_manager.progress.add_task(f"[cyan]Analyzing with {model_name}...", total=100)
        display_manager.add_debug(f"[cyan]Analyzing text with {model_name} - this may take several minutes...[/cyan]")
        start_time = time.time()
        
        try:
            # Try streaming approach for better monitoring
            analysis = ""
            tokens = 0
            
            # Create a monitor task with enhanced status information
            monitor_stop_event = asyncio.Event()
            monitor_task_id = display_manager.detail_progress.add_task(
                "[cyan]Connecting to Ollama...",  # Clear initial message
                total=None
            )
            monitor_task = asyncio.create_task(
                monitor_ollama_status(
                    monitor_task_id, 
                    display_manager.detail_progress, 
                    monitor_stop_event,
                    display_manager  # Pass display_manager to allow logging
                )
            )
            
            async with aiohttp.ClientSession() as session:
                data = {
                    'model': model_name,
                    'prompt': prompt,
                    'stream': True,
                    'options': {
                        'temperature': 0.2,
                        'num_predict': 4096
                    }
                }
                # Replace verbose output with concise summary
                display_manager.add_debug(f"[blue]Sending request to {model_name} - {prompt_size/1000:.1f}K chars{' (with custom prompt)' if has_custom_prompt else ''}[/blue]")
                async with session.post(
                    'http://localhost:11434/api/generate',
                    json=data,
                    timeout=aiohttp.ClientTimeout(total=1800)  # 30 minute timeout
                ) as response:
                    if response.status != 200:
                        raise ValueError(f"Ollama API error: {response.status}")
                    
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                
                                if 'response' in data:
                                    analysis += data['response']
                                    tokens += 1
                                    
                                    # Update progress every few tokens
                                    if tokens % 10 == 0:
                                        elapsed = time.time() - start_time
                                        rate = tokens / elapsed if elapsed > 0 else 0
                                        
                                        # Update status displays
                                        live.refresh()
                                        
                                        # Update progress
                                        display_manager.progress.update(
                                            analysis_task, 
                                            description=f"[cyan]Generated {tokens} tokens ({rate:.1f}/sec)",
                                            completed=min(100, tokens * 100 // 4096)  # Assume ~4K tokens total
                                        )
                                
                                # Show completion info
                                if 'done' in data and data['done']:
                                    if 'total_duration' in data:
                                        duration_sec = data['total_duration'] / 1_000_000_000
                                        display_manager.progress.update(
                                            analysis_task,
                                            description=f"[green]Analysis complete: {tokens} tokens in {duration_sec:.2f}s ({tokens/duration_sec:.1f}/s)",
                                            completed=100
                                        )
                                    else:
                                        display_manager.progress.update(analysis_task, completed=100, description=f"[green]Analysis complete: {tokens} tokens")
                                    break
                                    
                            except json.JSONDecodeError:
                                pass
            
            # Stop the monitoring task
            monitor_stop_event.set()
            try:
                await monitor_task
            except Exception as e:
                display_manager.add_debug(f"[yellow]Error stopping monitor: {str(e)}[/yellow]")
        
        except Exception as e:
            display_manager.add_debug(f"[yellow]Streaming error: {e}. Falling back to standard API.[/yellow]")
            # Fallback to standard API
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.2,
                    "num_predict": 4096
                }
            )
            analysis = response.message.content if hasattr(response, 'message') and hasattr(response.message, 'content') else response["message"]["content"]
            display_manager.progress.update(analysis_task, completed=100, description="[green]Analysis complete (non-streaming mode)")
        
        # Update final server status
        live.refresh()

        # Step 5: Save the results
        save_task = display_manager.progress.add_task("[cyan]Saving analysis results...", total=1)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"timeline_analysis_{timestamp}.md")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Message Timeline Analysis\n\n")
            f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} using {model_name}*\n\n")
            f.write(analysis)
            f.write(f"\n\n---\n\n*Analysis completed in {time.time() - start_time:.2f} seconds*\n\n")
            f.write("\n\n## Source Material\n\n")
            f.write(f"- [Raw OCR Data]({os.path.basename(results_file)})\n")
            f.write(f"- [Basic OCR Summary]({os.path.basename(markdown_path)})\n\n")

            if image_map:
                f.write("\n## Original Images\n\n")
                for i, (original_path, copied_path) in enumerate(image_map.items(), 1):
                    filename = os.path.basename(original_path)
                    f.write(f"### Image {i}: {filename}\n\n")
                    f.write(f"![Screenshot]({copied_path})\n\n")
                    
        display_manager.progress.update(save_task, completed=1, description=f"[green]Results saved to: {os.path.basename(output_path)}")
        live.refresh()

        # Clean up if we created our own display
        if using_local_display:
            display_task.cancel()
            try:
                await asyncio.wait_for(display_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            display_manager.stop()
            
        # Show final summary
        elapsed_time = time.time() - start_time
        console.print(f"[green]✓ Analysis completed in {elapsed_time:.2f} seconds[/green]")
        console.print(f"[green]✓ Generated approximately {tokens} tokens[/green]")
        console.print(f"[green]✓ Results saved to: {output_path}[/green]")

        # Open the file
        os.system(f'open "{output_path}"' if sys.platform == 'darwin' else 
                  f'xdg-open "{output_path}"' if sys.platform.startswith('linux') else 
                  f'start "" "{output_path}"')

    except ImportError:
        console.print("[red]Required packages not installed. Install with: pip install ollama aiohttp psutil[/red]")
    except Exception as e:
        console.print(f"[bold red]Error during analysis: {str(e)}[/bold red]")
        console.print(traceback.format_exc())
        if using_local_display:
            display_manager.stop()

def post_process_ocr_text(text: str) -> str:
    """Clean up OCR text for better readability"""
    replacements = {
        r'([0-9])l([0-9])': r'\1:\2',
        r'([0-9])I([0-9])': r'\1:\2',
        r'(< Message)': r'←Message',
        r'(< Back)': r'←Back',
        r',,': ',',
        r'\.\.\.\.': '...',
        r'\.\.\,': '...',
        r'``': '"',
        r"''": '"',
    }
    
    processed_text = text
    for pattern, replacement in replacements.items():
        processed_text = re.sub(pattern, replacement, processed_text)
    
    lines = processed_text.split('\n')
    filtered_lines = [line for line in lines if len(line.strip()) > 1 or line.strip().isdigit()]
    processed_text = '\n'.join(filtered_lines)
    processed_text = re.sub(r'\n\s*\n', '\n\n', processed_text)
    processed_text = re.sub(r' +', ' ', processed_text)
    
    return processed_text.strip()

def get_image_metadata(image_path: str) -> Dict[str, Any]:
    """Extract metadata from image"""
    try:
        file_stats = os.stat(image_path)
        file_name = os.path.basename(image_path)
        try:
            with Image.open(image_path) as img:
                width, height = img.size
        except Exception:
            width, height = None, None
        
        return {
            "filename": file_name,
            "path": image_path,
            "size_bytes": file_stats.st_size,
            "modified_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "created_time": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "width": width,
            "height": height
        }
    except Exception as e:
        return {
            "filename": os.path.basename(image_path),
            "path": image_path,
            "error": str(e)
        }

async def save_results(results: List[Dict[str, Any]], output_dir: str) -> str:
    """Save OCR results to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"ocr_results_{timestamp}.json")
    
    output_data = {
        "created_at": datetime.now().isoformat(),
        "file_count": len(results),
        "results": results
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_path

def display_result_summary(results: List[Dict[str, Any]]) -> Table:
    """Display summary of OCR results"""
    table = Table(title="OCR Results Summary")
    table.add_column("File", style="cyan")
    table.add_column("Text Length", style="green", justify="right")
    table.add_column("Preview", style="yellow")
    
    for result in results:
        filename = result.get("filename", "unknown")
        error = result.get("error")
        if error:
            table.add_row(filename, "ERROR", error)
        else:
            text = result.get("text", "")
            preview = text.replace("\n", " ")[:50] + "..." if len(text) > 50 else text.replace("\n", " ")
            table.add_row(filename, str(len(text)), preview)
    
    return table

def copy_images_to_output(file_paths: List[str], output_dir: str) -> Dict[str, str]:
    """Copy image files to output directory and return mapping"""
    image_map = {}
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    console.print("[cyan]Copying images to output directory...[/cyan]")
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        base, ext = os.path.splitext(filename)
        safe_filename = f"{base}_{int(time.time())}_{hash(file_path) % 10000:04d}{ext}"
        output_path = os.path.join(images_dir, safe_filename)
        
        try:
            import shutil
            shutil.copy2(file_path, output_path)
            image_map[file_path] = os.path.join("images", safe_filename)
        except Exception as e:
            console.print(f"[yellow]Error copying {filename}: {str(e)}[/yellow]")
    
    return image_map

async def create_markdown_summary(data: List[Dict[str, Any]], results_file: str, image_map: Dict[str, str] = None) -> str:
    """Create a markdown summary of the OCR results"""
    output_dir = os.path.dirname(results_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"ocr_summary_{timestamp}.md")
    
    for item in data:
        text = item.get("text", "")
        date_formats = [
            r'(\d{1,2}/\d{1,2}/\d{2,4})',
            r'(\d{1,2}-\d{1,2}-\d{2,4})',
            r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}[,]?\s+\d{2,4}',
            r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))',
            r'(\d{1,2}:\d{2})',
            r'(Today|Yesterday)',
        ]
        
        detected_dates = []
        for pattern in date_formats:
            matches = re.findall(pattern, text)
            if matches:
                detected_dates.extend(matches)
        
        if detected_dates:
            item["detected_dates"] = detected_dates
            item["date"] = detected_dates[0]
        
        names_pattern = r'(From|To|Sent by|Received from|Forwarded by):\s*([A-Za-z\s\.]+)'
        name_matches = re.findall(names_pattern, text)
        if name_matches:
            for match_type, name in name_matches:
                if match_type.lower().startswith('from') or match_type.lower().startswith('sent'):
                    item["sender"] = name.strip()
                elif match_type.lower().startswith('to'):
                    item["recipient"] = name.strip()
    
    markdown = [
        "# OCR Text Extraction Summary",
        f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Extracted Messages",
        ""
    ]
    
    for i, item in enumerate(data, 1):
        filename = item.get("filename", "unknown")
        file_path = item.get("file_path", "")
        text = item.get("text", "").strip()
        date_info = item.get("date", "Unknown date")
        sender_info = item.get("sender", "Unknown sender")
        
        header = f"### Message {i}"
        if date_info != "Unknown date":
            header += f": {date_info}"
            
        markdown.append(header)
        markdown.append("")
        
        markdown.append("| Field | Value |")
        markdown.append("|-------|-------|")
        markdown.append(f"| Source file | `{filename}` |")
        markdown.append(f"| Sender | {sender_info} |")
        if "detected_dates" in item:
            markdown.append(f"| Detected dates | {', '.join(item['detected_dates'][:3])} |")
        markdown.append("")
        
        if image_map and file_path in image_map:
            rel_path = image_map[file_path]
            markdown.append(f"![Screenshot]({rel_path})")
            markdown.append("")
        
        markdown.append("<details>")
        markdown.append("<summary>View extracted text</summary>")
        markdown.append("")
        markdown.append("```")
        markdown.append(text)
        markdown.append("```")
        markdown.append("</details>")
        markdown.append("")
        
        markdown.append("---")
        markdown.append("")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown))
    
    return output_path

def get_full_prompt(data: List[Dict[str, Any]]) -> str:
    """Generate the full prompt that would be sent to the model"""
    simplified_data = []
    for result in data:
        simplified_entry = {
            "filename": result.get("filename", "unknown"),
            "text": result.get("text", ""),
        }
        if "date" in result:
            simplified_entry["date"] = result["date"]
        if "sender" in result:
            simplified_entry["sender"] = result["sender"]
        simplified_data.append(simplified_entry)
    
    post_prompt = ""
    post_prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "postprompt.txt")
    if os.path.exists(post_prompt_file):
        try:
            with open(post_prompt_file, 'r', encoding='utf-8') as f:
                post_prompt = f.read()
        except Exception as e:
            console.print(f"[yellow]Error reading postprompt.txt: {str(e)}[/yellow]")
    
    prompt = ANALYSIS_PROMPT + "\n\nHere's the OCR data extracted from screenshots:\n"
    prompt += json.dumps(simplified_data, indent=2)
    
    if post_prompt:
        prompt += "\n\n" + post_prompt
    
    return prompt

def display_full_prompt() -> None:
    """Display the full prompt that would be sent to the model"""
    try:
        # Use the selected directory from config instead of current working directory
        config = load_config()
        selected_dir = config.get("last_directory", os.getcwd())
        
        console.print(f"[cyan]Looking for images in: {selected_dir}[/cyan]")
        file_paths = get_all_images_in_directory(selected_dir)
        
        if not file_paths:
            console.print("[yellow]No supported image files found in the selected directory.[/yellow]")
            console.print("[yellow]Supported extensions: {0}[/yellow]".format(", ".join(SUPPORTED_EXTENSIONS)))
            return
        
        console.print(f"[green]Found {len(file_paths)} image files.[/green]")
        mock_results = [{"filename": os.path.basename(path), "text": "Sample OCR text for preview"} for path in file_paths[:3]]
        prompt = get_full_prompt(mock_results)
        
        console.print("[bold cyan]Full Prompt Preview:[/bold cyan]")
        console.print(Panel(prompt, width=100, border_style="green"))
        
        post_prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "postprompt.txt")
        if not os.path.exists(post_prompt_file):
            console.print("\n[yellow]Note: 'postprompt.txt' not found in the application directory:[/yellow]")
            console.print(f"[yellow]{os.path.dirname(os.path.abspath(__file__))}[/yellow]")
            console.print("[yellow]Create this file to add custom instructions to the prompt.[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]Error displaying prompt: {str(e)}[/bold red]")

def config_menu(config: Dict[str, Any]) -> Dict[str, Any]:
    """Display and handle the configuration menu"""
    while True:
        console.print("\n[bold cyan]Configuration Menu:[/bold cyan]")
        console.print("  1. Select input directory")
        console.print("  2. Change output directory")
        console.print("  3. View full prompt")
        console.print("  4. View available models")
        console.print("  5. Select model")
        console.print("  6. Back to main menu")
        
        choice = Prompt.ask("\nChoose an option", choices=["1", "2", "3", "4", "5", "6"], default="6")
        
        if choice == "1":
            prev_dir = config.get("last_directory", os.getcwd())
            new_dir = select_directory()
            if (prev_dir != new_dir):
                config["last_directory"] = new_dir
                save_config(config)
        
        elif choice == "2":
            prev_output_dir = config.get("output_directory", DEFAULT_OUTPUT_DIR)
            console.print(f"[cyan]Current output directory: {prev_output_dir}[/cyan]")
            console.print("[yellow]Enter new output directory path (or leave empty to keep current):[/yellow]")
            new_output_dir = input("> ").strip()
            
            if new_output_dir:
                new_output_dir = os.path.expanduser(new_output_dir)
                try:
                    os.makedirs(new_output_dir, exist_ok=True)
                    config["output_directory"] = new_output_dir
                    save_config(config)
                    console.print(f"[green]Output directory updated to: {new_output_dir}[/green]")
                except Exception as e:
                    console.print(f"[red]Error creating directory: {str(e)}[/red]")
        
        elif choice == "3":
            display_full_prompt()
            console.print("[blue]Note: Custom prompt extensions should be placed in 'postprompt.txt' in the application folder.[/blue]")
        
        elif choice == "4":
            models = get_available_models()
            if models:
                console.print("[blue]Available models:[/blue]")
                for model in models:
                    console.print(f"  - {model}")
            else:
                console.print("[red]No models available or Ollama not reachable.[/red]")
        
        elif choice == "5":
            models = get_available_models()
            if models:
                console.print("[blue]Available models:[/blue]")
                for model in models:
                    console.print(f"  - {model}")
                selected_model = Prompt.ask("Enter the model name to select", choices=models, default=config.get("selected_model", "yarn-llama2"))
                config["selected_model"] = selected_model
                save_config(config)
                console.print(f"[green]Model updated to {selected_model}.[/green]")
            else:
                console.print("[red]No models available or Ollama not reachable.[/red]")
        
        elif choice == "6":
            break
    
    return config

def get_available_models() -> List[str]:
    """Get available models from Ollama with improved error handling"""
    try:
        import ollama
        models = None
        
        try:
            models = ollama.list()
        except Exception as e:
            console.print(f"[yellow]Error listing models: {str(e)}[/yellow]")
            return []
            
        model_names = []
        
        # Handle different response formats
        if isinstance(models, list):
            model_names = [m.model for m in models if hasattr(m, 'model')]
        elif hasattr(models, 'models') and isinstance(models.models, list):
            model_names = [m.model for m in models.models if hasattr(m, 'model')]
        elif isinstance(models, dict) and "models" in models and isinstance(models["models"], list):
            for m in models["models"]:
                if isinstance(m, dict) and "model" in m:
                    model_names.append(m["model"])
                elif hasattr(m, 'model'):
                    model_names.append(m.model)
        
        if model_names:
            return model_names
        
        # Try alternate approach with CLI if Python API returns no models
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout:
                lines = result.stdout.strip().split('\n')
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        parts = line.split()
                        if parts:
                            model_names.append(parts[0])
        except Exception as e:
            console.print(f"[yellow]CLI fallback error: {str(e)}[/yellow]")
        
        return model_names
    except ImportError:
        console.print("[yellow]Ollama Python package not installed. Install with: pip install ollama[/yellow]")
        return []
    except Exception as e:
        console.print(f"[yellow]Error fetching models: {str(e)}[/yellow]")
        return []

def show_configuration_status(config: Dict[str, Any]) -> None:
    """Display the current configuration status"""
    config_table = Table(title="Current Configuration", box=None, padding=(0, 1))
    
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value", style="green")
    config_table.add_column("Status", style="yellow")
    
    # Input directory
    input_dir = config.get("last_directory", os.getcwd())
    file_count = len(get_all_images_in_directory(input_dir))
    input_status = f"{file_count} images found" if file_count > 0 else "No images found"
    config_table.add_row("Input directory", input_dir, input_status)
    
    # Output directory
    output_dir = config.get("output_directory", DEFAULT_OUTPUT_DIR)
    output_status = "Ready" if os.path.isdir(output_dir) else "Will be created"
    config_table.add_row("Output directory", output_dir, output_status)
    
    # Model
    model = config.get("selected_model", "yarn-llama2")
    model_status = ""
    try:
        import ollama
        models = get_available_models()
        if model in models:
            model_status = "Available"
        else:
            model_status = "Not installed (will be pulled)"
    except:
        model_status = "Ollama status unknown"
    config_table.add_row("Selected model", model, model_status)
    
    # OCR language
    lang = config.get("last_language", "eng")
    config_table.add_row("OCR language", lang, "")
    
    # Custom prompt
    post_prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "postprompt.txt")
    if os.path.exists(post_prompt_file):
        try:
            with open(post_prompt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                status = "Custom prompt loaded" if content.strip() else "Empty file"
        except:
            status = "Error reading file"
    else:
        status = "No custom prompt"
    config_table.add_row("Custom prompt", post_prompt_file, status)
    
    console.print(config_table)

async def main():
    try:
        console.print(banner())
        check_tesseract_installation()
        check_ollama_status()  # Add Ollama status check
        
        config = load_config()
        current_directory = config.get("last_directory", os.getcwd())
        output_dir = config.get("output_directory", DEFAULT_OUTPUT_DIR)
        lang = config.get("last_language", "eng")
        
        # Create a single display manager for the entire process
        display_manager = DisplayManager()
        
        while True:
            console.print("\n[bold cyan]Menu Options:[/bold cyan]")
            console.print("  1. Process all images in directory")
            console.print("  2. Configuration")
            console.print("  3. Exit")
            
            # Show current configuration status
            show_configuration_status(config)
            
            choice = Prompt.ask("\nChoose an option", choices=["1", "2", "3"], default="1")
            
            if choice == "1":
                console.print(f"\n[bold]Processing images in: {current_directory}[/bold]")
                
                if not os.path.isdir(current_directory):
                    console.print("[red]Directory doesn't exist![/red]")
                    continue
                    
                file_paths = get_all_images_in_directory(current_directory)
                if not file_paths:
                    console.print("[yellow]No supported image files found in this directory.[/yellow]")
                    continue
                
                # Show processing details
                model_name = get_current_model()
                post_prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "postprompt.txt")
                has_custom_prompt = os.path.exists(post_prompt_file) and os.path.getsize(post_prompt_file) > 0
                
                console.print(Panel.fit(
                    f"[bold green]Ready to process {len(file_paths)} images[/bold green]\n"
                    f"Using model: [cyan]{model_name}[/cyan]\n"
                    f"Custom prompt: {'[green]Yes[/green]' if has_custom_prompt else '[yellow]No[/yellow]'}\n"
                    f"Output directory: [blue]{output_dir}[/blue]",
                    title="Processing Summary",
                    border_style="green"
                ))
                
                if not Confirm.ask("Continue with processing?", default=True):
                    console.print("[yellow]Processing cancelled.[/yellow]")
                    continue
                
                with display_manager.start() as live:
                    display_task = await display_manager.start_display_updater()
                    
                    try:
                        # Process images with shared display manager
                        results = await process_images(file_paths, lang, display_manager)
                        
                        if not results:
                            display_manager.add_debug("[yellow]No results extracted.[/yellow]")
                            continue
                        
                        # Save results
                        results_file = await save_results(results, output_dir)
                        display_manager.add_debug(f"[green]OCR results saved to: {results_file}[/green]")
                        
                        # Continue to analysis with the same display manager
                        await analyze_with_ollama(results_file, output_dir, display_manager)
                    finally:
                        # Clean up display
                        display_task.cancel()
                        try:
                            await asyncio.wait_for(display_task, timeout=1.0)
                        except (asyncio.CancelledError, asyncio.TimeoutError):
                            pass
            
            elif choice == "2":
                config = config_menu(config)
                current_directory = config.get("last_directory", os.getcwd())
                output_dir = config.get("output_directory", DEFAULT_OUTPUT_DIR)
            
            elif choice == "3":
                console.print("[cyan]Exiting application.[/cyan]")
                
                config["last_directory"] = current_directory
                config["output_directory"] = output_dir
                config["last_language"] = lang
                save_config(config)
                
                return
                
            if choice != "3":
                console.print("\nPress Enter to continue...")
                try:
                    input()
                except Exception:
                    pass

    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user. Exiting...[/yellow]")
        
        try:
            config = load_config()
            if 'current_directory' in locals():
                config["last_directory"] = current_directory
            if 'output_dir' in locals():
                config["output_directory"] = output_dir
            if 'lang' in locals():
                config["last_language"] = lang
            save_config(config)
        except Exception as e:
            console.print(f"[yellow]Error saving config during exit: {e}[/yellow]")
        
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())
        
        console.print("[yellow]Press Enter to exit...[/yellow]")
        try:
            input()
        except:
            pass

if __name__ == "__main__":
    try:
        if sys.platform == 'win32':
            import asyncio
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            asyncio.run(main())
        else:
            import asyncio
            asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


