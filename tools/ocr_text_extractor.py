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

# Constants
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.JPG', '.PNG', '.JPEG', '.TIFF', '.pdf', '.PDF'}
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/ocr_results")
CONFIG_DIR = os.path.expanduser("~/.ocr_extractor")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

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
        "selected_model": "yarn-llama2"
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
        selected_model = config.get("selected_model", "yarn-llama2:latest")
        
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

async def clean_text_with_ollama(text: str, progress: Optional[Progress] = None, task_id: Optional[int] = None) -> str:
    """Clean the OCR text using Ollama."""
    try:
        import ollama
    except ImportError:
        console.print("[yellow]Ollama not installed. Install with: pip install ollama[/yellow]")
        return text
        
    try:
        config = load_config()
        model_name = config.get("selected_model", "yarn-llama2")
        
        if progress:
            progress.update(task_id, description=f"[magenta]Ollama: Cleaning text using {model_name}...")
        
        prompt = f"Correct and clean the following OCR text and do not provide any preamble in your response, only the cleaned up text.:\n\n{text}\n\nCleaned Text:"
        
        # Create a monitoring task for this operation
        monitor_stop_event = None
        monitoring_task = None
        
        if progress:
            monitor_task_id = progress.add_task(f"[cyan]Monitoring Ollama...", total=None, visible=True)
            monitor_stop_event = asyncio.Event()
            monitoring_task = asyncio.create_task(
                monitor_ollama_status(monitor_task_id, progress, monitor_stop_event)
            )
        
        try:
            # Try streaming approach first for better monitoring
            cleaned_text = await stream_ollama_completion(
                model_name, 
                prompt, 
                temperature=0.1, 
                progress=progress, 
                task_id=task_id
            )
        except Exception as e:
            console.print(f"[yellow]Streaming approach failed: {str(e)}. Falling back to standard API.[/yellow]")
            # Fallback to standard API with proper error handling
            try:
                response = ollama.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.1}
                )
                # Handle both possible response formats
                if hasattr(response, 'message') and hasattr(response.message, 'content'):
                    cleaned_text = response.message.content
                elif isinstance(response, dict) and "message" in response:
                    if isinstance(response["message"], dict) and "content" in response["message"]:
                        cleaned_text = response["message"]["content"]
                    else:
                        console.print("[yellow]Unexpected response format from Ollama. Using original text.[/yellow]")
                        cleaned_text = text
                else:
                    console.print("[yellow]Unexpected response format from Ollama. Using original text.[/yellow]")
                    cleaned_text = text
            except Exception as inner_e:
                console.print(f"[yellow]Standard API also failed: {str(inner_e)}. Using original text.[/yellow]")
                cleaned_text = text
        finally:
            # Stop monitoring if active
            if monitor_stop_event:
                monitor_stop_event.set()
                if monitoring_task:
                    try:
                        await asyncio.wait_for(monitoring_task, timeout=5.0)
                    except (asyncio.CancelledError, asyncio.TimeoutError):
                        pass
                    except Exception as e:
                        console.print(f"[yellow]Error stopping monitoring: {str(e)}[/yellow]")
                
                # Hide the monitoring task
                if progress:
                    progress.update(monitor_task_id, visible=False)
                    
        return cleaned_text
    except Exception as e:
        console.print(f"[yellow]Error cleaning text with Ollama: {str(e)}[/yellow]")
        return text

async def stream_ollama_completion(model: str, prompt: str, temperature: float = 0.2, progress: Optional[Progress] = None, task_id: Optional[int] = None) -> str:
    """Stream a completion from Ollama with detailed status updates"""
    try:
        result = ""
        tokens = 0
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            try:
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
                    timeout=aiohttp.ClientTimeout(total=300)  # 5 minute timeout
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Ollama API error: {response.status} - {error_text}")
                        
                    async for line in response.content:
                        if line:
                            try:
                                data = json.loads(line)
                                
                                if 'response' in data:
                                    result += data['response']
                                    tokens += 1
                                    
                                    if progress and task_id and tokens % 5 == 0:
                                        elapsed = time.time() - start_time
                                        rate = tokens / elapsed if elapsed > 0 else 0
                                        progress.update(
                                            task_id, 
                                            description=f"[magenta]Ollama: Generated {tokens} tokens ({rate:.1f}/sec)"
                                        )
                                
                                # Update with completion statistics when done
                                if data.get('done', False):
                                    if progress and task_id and 'total_duration' in data:
                                        duration_sec = data['total_duration'] / 1_000_000_000  # ns to s
                                        progress.update(
                                            task_id,
                                            description=f"[green]Ollama: Completed ({tokens} tokens, {duration_sec:.2f}s)"
                                        )
                                    break
                                    
                            except json.JSONDecodeError:
                                pass
            except aiohttp.ClientConnectorError as e:
                raise ValueError(f"Could not connect to Ollama server: {str(e)}")
        
        return result
    except Exception as e:
        raise ValueError(f"Error streaming from Ollama: {str(e)}")

async def process_images(file_paths: List[str], lang: str = "eng") -> List[Dict[str, Any]]:
    """Process images and extract OCR text with detailed status"""
    if not file_paths:
        console.print("[yellow]No files selected for processing.[/yellow]")
        return []
    
    results = []
    
    # Create a layout for a more sophisticated display
    layout = Layout()
    layout.split(
        Layout(name="main", ratio=3),
        Layout(name="status", ratio=1)
    )
    
    # Progress display for main tasks
    main_progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green"),
        TextColumn("[cyan]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )
    
    # Status table to show system info
    status_table = Table.grid()
    status_table.add_column("Component", style="blue")
    status_table.add_column("Status", style="green")
    
    # Initialize the status table with some values right away
    update_status_table(status_table, results, file_paths)
    
    # Configure the layout
    layout["main"].update(main_progress)
    layout["status"].update(Panel(status_table, title="System Status", border_style="blue"))
    
    # Main task progress
    with Live(layout, refresh_per_second=4) as live:
        main_task = main_progress.add_task(f"[cyan]Processing {len(file_paths)} images...", total=len(file_paths))
        
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            main_progress.update(main_task, description=f"[cyan]Processing {filename}...")
            
            # Add subtasks for better tracking
            ocr_task = main_progress.add_task(f"[blue]  OCR: {filename}", total=1, visible=True)
            
            try:
                ocr_text = await extract_text_from_image(file_path, lang, main_progress, ocr_task)
                main_progress.update(ocr_task, completed=1, description=f"[green]  OCR: Completed {filename}")
                
                metadata = get_image_metadata(file_path)
                
                result = {
                    "filename": filename,
                    "file_path": file_path,
                    "metadata": metadata,
                    "text": ocr_text,
                    "processed_at": datetime.now().isoformat()
                }
                
                results.append(result)
                main_progress.update(main_task, advance=1)
                
            except Exception as e:
                console.print(f"[bold red]Error processing {filename}: {str(e)}[/bold red]")
                main_progress.update(ocr_task, visible=False)
                results.append({
                    "filename": filename,
                    "file_path": file_path,
                    "error": str(e),
                    "processed_at": datetime.now().isoformat()
                })
                main_progress.update(main_task, advance=1)
            
            # Update status table after each file - passing needed parameters
            update_status_table(status_table, results, file_paths)
            live.refresh()
    
    return results

async def monitor_ollama_status(task_id, progress, stop_event):
    """Monitor the Ollama server status with comprehensive metrics"""
    try:
        import psutil
        import aiohttp
        
        while not stop_event.is_set():
            stats = {}
            
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
                        stats["CPU"] = f"{proc.cpu_percent():.1f}%"
                        if 'memory_info' in proc_info and proc_info['memory_info']:
                            stats["Memory"] = f"{proc_info['memory_info'].rss / (1024 * 1024):.1f} MB"
                        break
                
                # Only try API if process wasn't found
                if not found:
                    stats["Process"] = "Not detected"
            except Exception as e:
                stats["Process Error"] = str(e)[:30]
                
            # Try metrics endpoint for advanced metrics with better error handling
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get('http://localhost:11434/api/metrics', timeout=2) as response:
                        if response.status == 200:
                            text = await response.text()
                            # Parse Prometheus format metrics
                            if "ollama_requests_total" in text:
                                for line in text.split('\n'):
                                    if line and not line.startswith('#'):
                                        if "ollama_tokens_total" in line and "type=\"completion\"" in line:
                                            try:
                                                value = float(line.split('}')[1].strip())
                                                stats["Tokens"] = f"{int(value)}"
                                            except:
                                                pass
                                        elif "ollama_requests_total" in line and "status=\"200\"" in line:
                                            try:
                                                value = float(line.split('}')[1].strip())
                                                stats["Requests"] = f"{int(value)}"
                                            except:
                                                pass
            except aiohttp.ClientConnectorError:
                stats["API"] = "Not available"
            except Exception as e:
                stats["API Error"] = str(e)[:30]
                
            # Always update with some status info even if empty
            if not stats:
                stats["Status"] = "Checking..."
                
            # Update progress display
            status_text = " | ".join([f"{k}: {v}" for k, v in stats.items()])
            progress.update(task_id, description=f"[cyan]Ollama: {status_text}")
            
            # Check stop event with a timeout
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                pass
            
    except Exception as e:
        progress.update(task_id, description=f"[red]Monitor error: {str(e)}")
    finally:
        # Tasks can't be removed, just hide it
        progress.update(task_id, visible=False)

async def analyze_with_ollama(results_file: str, output_dir: str):
    """Analyze OCR results with Ollama using the selected model with detailed monitoring"""
    try:
        import ollama
        import aiohttp
        import json
        
        # Set up a layout for a sophisticated display with multiple panels
        layout = Layout()
        layout.split_column(
            Layout(name="progress", ratio=4),
            Layout(name="server_status", ratio=1)
        )
        
        # Progress tracking
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="cyan", finished_style="green"),
            TimeElapsedColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )
        
        # Server status table
        server_table = Table.grid(padding=(0, 1))
        server_table.add_column("Component", style="blue")
        server_table.add_column("Value", style="green")
        
        # Function to update server status
        def update_server_status(tokens=None, rate=None):
            server_table.rows = []
            
            # Add ollama process info
            for proc in psutil.process_iter(['name', 'cpu_percent', 'memory_info']):
                if 'ollama' in proc.info['name'].lower():
                    server_table.add_row("Ollama CPU", f"{proc.info['cpu_percent']:.1f}%")
                    server_table.add_row("Ollama Memory", f"{proc.info['memory_info'].rss / (1024 * 1024):.1f} MB")
                    break
            
            # Add token generation stats
            if tokens is not None:
                server_table.add_row("Tokens Generated", f"{tokens}")
            if rate is not None:
                server_table.add_row("Generation Rate", f"{rate:.2f} tokens/sec")
            
            # Add system stats
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            server_table.add_row("System CPU", f"{cpu_percent}%")
            server_table.add_row("System Memory", f"{memory.percent}%")
            
            # Try to get GPU info if available
            try:
                result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                                      '--format=csv,noheader,nounits'], 
                                      capture_output=True, text=True, timeout=1)
                if result.returncode == 0:
                    values = result.stdout.strip().split(',')
                    server_table.add_row("GPU Usage", f"{values[0].strip()}%")
                    server_table.add_row("GPU Memory", f"{values[1].strip()} MB")
            except:
                pass
        
        # Configure layout components
        layout["progress"].update(progress)
        layout["server_status"].update(Panel(server_table, title="Ollama Server Status", border_style="blue"))
        
        update_server_status()  # Initial update
        
        with Live(layout, refresh_per_second=4) as live:
            console.print("[cyan]Connecting to Ollama server...[/cyan]")
            
            # Load results
            load_task = progress.add_task("[cyan]Loading OCR results...", total=1)
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                results = data.get("results", [])
            progress.update(load_task, completed=1)

            file_paths = [result.get("file_path") for result in results if "file_path" in result]
            
            # Step 1: Create summary and copy images
            copy_task = progress.add_task("[cyan]Creating output resources...", total=2)
            
            progress.update(copy_task, description="[cyan]Copying images to output directory...")
            image_map = copy_images_to_output(file_paths, output_dir)
            progress.update(copy_task, advance=1)
            live.refresh()

            progress.update(copy_task, description="[cyan]Creating markdown summary...")
            markdown_path = await create_markdown_summary(results, results_file, image_map)
            progress.update(copy_task, advance=1, description="[green]Output resources prepared")
            console.print(f"[green]✓ Created initial markdown summary: {markdown_path}[/green]")
            live.refresh()

            # Step 2: Prepare the prompt
            prompt_task = progress.add_task("[cyan]Building analysis prompt...", total=1)
            
            # Read postprompt.txt if it exists
            post_prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "postprompt.txt")
            post_prompt = ""
            has_custom_prompt = False
            if os.path.exists(post_prompt_file):
                try:
                    with open(post_prompt_file, 'r', encoding='utf-8') as f:
                        post_prompt = f.read()
                    if post_prompt.strip():
                        has_custom_prompt = True
                        console.print(f"[green]✓ Loaded custom prompt from {post_prompt_file}[/green]")
                except Exception as e:
                    console.print(f"[yellow]Error reading postprompt.txt: {str(e)}[/yellow]")
            else:
                console.print(f"[blue]Using default prompt only (no postprompt.txt found)[/blue]")

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
            progress.update(prompt_task, completed=1, description=f"[green]Prompt ready: {prompt_size/1000:.1f}K chars (~{token_estimate} tokens)")
            live.refresh()
            
            # Step 3: Set up the model
            config = load_config()
            model_name = config.get("selected_model", "yarn-llama2")
            model_task = progress.add_task(f"[cyan]Setting up model: {model_name}...", total=1)

            try:
                models = ollama.list()
                model_names = [m.model for m in models if hasattr(m, 'model')]

                if model_name not in model_names:
                    progress.update(model_task, description=f"[yellow]Pulling model: {model_name}...")
                    ollama.pull(model_name)
                progress.update(model_task, completed=1, description=f"[green]Model {model_name} ready")
            except Exception as e:
                console.print(f"[yellow]Error checking models, will try to use {model_name} directly: {str(e)}[/yellow]")
                progress.update(model_task, completed=1, description=f"[yellow]Model status unknown")
            
            live.refresh()

            # Step 4: Run the analysis with detailed monitoring
            analysis_task = progress.add_task(f"[cyan]Analyzing with {model_name}...", total=100)
            console.print(f"[cyan]Analyzing text with {model_name} - this may take several minutes...[/cyan]")
            start_time = time.time()
            
            try:
                # Try streaming approach for better monitoring
                analysis = ""
                tokens = 0
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        'http://localhost:11434/api/generate',
                        json={
                            'model': model_name,
                            'prompt': prompt,
                            'stream': True,
                            'options': {
                                'temperature': 0.2,
                                'num_predict': 4096
                            }
                        },
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
                                            update_server_status(tokens, rate)
                                            live.refresh()
                                            
                                            # Update progress
                                            progress.update(
                                                analysis_task, 
                                                description=f"[cyan]Generated {tokens} tokens ({rate:.1f}/sec)",
                                                completed=min(100, tokens * 100 // 4096)  # Assume ~4K tokens total
                                            )
                                    
                                    # Show completion info
                                    if 'done' in data and data['done']:
                                        if 'total_duration' in data:
                                            duration_sec = data['total_duration'] / 1_000_000_000
                                            progress.update(
                                                analysis_task,
                                                description=f"[green]Analysis complete: {tokens} tokens in {duration_sec:.2f}s ({tokens/duration_sec:.1f}/s)",
                                                completed=100
                                            )
                                        else:
                                            progress.update(analysis_task, completed=100, description=f"[green]Analysis complete: {tokens} tokens")
                                        break
                                        
                                except json.JSONDecodeError:
                                    pass
                
            except Exception as e:
                console.print(f"[yellow]Streaming error: {e}. Falling back to standard API.[/yellow]")
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
                progress.update(analysis_task, completed=100, description="[green]Analysis complete (non-streaming mode)")
            
            # Update final server status
            update_server_status()
            live.refresh()

            # Step 5: Save the results
            save_task = progress.add_task("[cyan]Saving analysis results...", total=1)
            
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
                        
            progress.update(save_task, completed=1, description=f"[green]Results saved to: {os.path.basename(output_path)}")
            live.refresh()

        # Show final summary outside of Live display
        elapsed_time = time.time() - start_time
        console.print(f"[green]✓ Analysis completed in {elapsed_time:.2f} seconds[/green]")
        console.print(f"[green]✓ Generated approximately {tokens} tokens[/green]")
        console.print(f"[green]✓ Results saved to: {output_path}[/green]")

        # Open the file if on macOS/Linux
        os.system(f'open "{output_path}"' if sys.platform == 'darwin' else f'xdg-open "{output_path}"' if sys.platform.startswith('linux') else f'start "" "{output_path}"')

    except ImportError:
        console.print("[red]Required packages not installed. Install with: pip install ollama aiohttp psutil[/red]")
    except Exception as e:
        console.print(f"[bold red]Error during analysis: {str(e)}[/bold red]")
        console.print(traceback.format_exc())

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
            if prev_dir != new_dir:
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

async def analyze_with_ollama(results_file: str, output_dir: str):
    """Analyze OCR results with Ollama using the selected model"""
    try:
        import ollama
        import aiohttp
        import json

        console.print("[cyan]Connecting to Ollama server...[/cyan]")

        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results = data.get("results", [])

        file_paths = [result.get("file_path") for result in results if "file_path" in result]
        
        # Create progress display for multiple steps
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="cyan", finished_style="green"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            # Step 1: Create summary and copy images
            copy_task = progress.add_task("[cyan]Creating output resources...", total=2)
            
            progress.update(copy_task, description="[cyan]Copying images to output directory...")
            image_map = copy_images_to_output(file_paths, output_dir)
            progress.update(copy_task, advance=1)

            progress.update(copy_task, description="[cyan]Creating markdown summary...")
            markdown_path = await create_markdown_summary(results, results_file, image_map)
            progress.update(copy_task, advance=1, description="[green]Resources created")
            console.print(f"[green]✓ Created initial markdown summary: {markdown_path}[/green]")

            # Step 2: Prepare the prompt
            prompt_task = progress.add_task("[cyan]Building analysis prompt...", total=1)
            
            # Read postprompt.txt if it exists
            post_prompt_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "postprompt.txt")
            post_prompt = ""
            has_custom_prompt = False
            if os.path.exists(post_prompt_file):
                try:
                    with open(post_prompt_file, 'r', encoding='utf-8') as f:
                        post_prompt = f.read()
                    if post_prompt.strip():
                        has_custom_prompt = True
                        console.print(f"[green]✓ Loaded custom prompt from {post_prompt_file}[/green]")
                except Exception as e:
                    console.print(f"[yellow]Error reading postprompt.txt: {str(e)}[/yellow]")
            else:
                console.print(f"[blue]Using default prompt only (no postprompt.txt found)[/blue]")

            # Generate the prompt with simplified data and post-prompt
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
            
            progress.update(prompt_task, completed=1, description="[green]Prompt prepared")
            
            # Step 3: Set up the model
            config = load_config()
            model_name = config.get("selected_model", "yarn-llama2")
            model_task = progress.add_task(f"[cyan]Setting up model: {model_name}...", total=1)

            try:
                models = ollama.list()
                model_names = [m.model for m in models if hasattr(m, 'model')]

                if model_name not in model_names:
                    progress.update(model_task, description=f"[yellow]Pulling {model_name}...")
                    ollama.pull(model_name)
                progress.update(model_task, completed=1, description=f"[green]Model {model_name} ready")
            except Exception as e:
                console.print(f"[yellow]Error checking models, will try to use {model_name} directly: {str(e)}[/yellow]")
                progress.update(model_task, completed=1, description=f"[yellow]Model status unknown")

            # Step 4: Monitor the Ollama server in a separate task
            monitor_task = progress.add_task("[cyan]Monitoring Ollama...", total=None)
            
            # Start a separate task to monitor Ollama status
            monitor_stop_event = asyncio.Event()
            monitoring_task = asyncio.create_task(
                monitor_ollama_status(monitor_task, progress, monitor_stop_event)
            )

            # Step 5: Run the analysis
            console.print(f"[cyan]Analyzing text with {model_name}...[/cyan]")
            start_time = time.time()
            analysis_task = progress.add_task(f"[cyan]Analyzing with {model_name}...", total=100)

            # Try to use streaming if available
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        'http://localhost:11434/api/generate',
                        json={
                            'model': model_name,
                            'prompt': prompt,
                            'stream': True,
                            'options': {
                                'temperature': 0.2,
                                'num_predict': 4096
                            }
                        },
                        timeout=aiohttp.ClientTimeout(total=1200)  # 20 minute timeout
                    ) as response:
                        if response.status != 200:
                            # Fallback to non-streaming approach if streaming fails
                            raise ValueError("Streaming not supported or failed")
                            
                        analysis = ""
                        tokens = 0
                        start_time = time.time()
                        
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
                                            remaining = int(4000 / rate) - int(elapsed) if rate > 0 else "?"
                                            progress.update(
                                                analysis_task, 
                                                description=f"[cyan]Generated {tokens} tokens ({rate:.1f}/sec)",
                                                completed=min(100, tokens * 100 // 4000)
                                            )
                                    
                                    if 'done' in data and data['done']:
                                        progress.update(analysis_task, completed=100, description=f"[green]Analysis complete ({tokens} tokens)")
                                        break
                                        
                                except json.JSONDecodeError:
                                    pass
                            
            except Exception as e:
                console.print(f"[yellow]Streaming error: {e}. Falling back to standard API.[/yellow]")
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
                progress.update(analysis_task, completed=100, description="[green]Analysis complete")

            # Stop the monitoring task
            monitor_stop_event.set()
            await monitoring_task

            # Step 6: Save the results
            save_task = progress.add_task("[cyan]Saving analysis results...", total=1)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"timeline_analysis_{timestamp}.md")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("# Message Timeline Analysis\n\n")
                f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} using {model_name}*\n\n")
                f.write(analysis)
                f.write("\n\n## Source Material\n\n")
                f.write(f"- [Raw OCR Data]({os.path.basename(results_file)})\n")
                f.write(f"- [Basic OCR Summary]({os.path.basename(markdown_path)})\n\n")

                if image_map:
                    f.write("\n## Original Images\n\n")
                    for i, (original_path, copied_path) in enumerate(image_map.items(), 1):
                        filename = os.path.basename(original_path)
                        f.write(f"### Image {i}: {filename}\n\n")
                        f.write(f"![Screenshot]({copied_path})\n\n")
                        
            progress.update(save_task, completed=1, description="[green]Results saved")

        elapsed_time = time.time() - start_time
        console.print(f"[green]✓ Analysis completed in {elapsed_time:.2f} seconds[/green]")
        console.print(f"[green]✓ Results saved to: {output_path}[/green]")

        os.system(f'open "{output_path}"')

    except ImportError:
        console.print("[red]Required packages not installed. Install with: pip install ollama aiohttp[/red]")
    except Exception as e:
        console.print(f"[bold red]Error during analysis: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

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

def update_status_table(status_table, results, file_paths):
    """Update the status table with current system info"""
    # Clear the table
    status_table.rows = []
    
    # Add CPU and RAM information
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        status_table.add_row("CPU Usage", f"{cpu_percent}%")
        status_table.add_row("Memory Usage", f"{memory.percent}% ({memory.used // (1024*1024)} MB / {memory.total // (1024*1024)} MB)")
    except Exception as e:
        status_table.add_row("System Stats", f"Error: {str(e)[:30]}")
    
    # Add Ollama process info if running
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_info']):
            proc_info = proc.info
            if 'name' in proc_info and proc_info['name'] and 'ollama' in proc_info['name'].lower():
                # Try to get accurate CPU usage
                proc.cpu_percent()
                time.sleep(0.1)  # Brief pause for CPU usage calculation to update
                current_cpu = proc.cpu_percent()
                
                if 'memory_info' in proc_info and proc_info['memory_info']:
                    status_table.add_row("Ollama CPU", f"{current_cpu:.1f}%")
                    status_table.add_row("Ollama Memory", f"{proc_info['memory_info'].rss / (1024 * 1024):.1f} MB")
                    break
    except Exception as e:
        status_table.add_row("Ollama Status", f"Error: {str(e)[:30]}")
    
    # Add file processing stats
    try:
        processed_count = len([r for r in results if 'error' not in r])
        error_count = len([r for r in results if 'error' in r])
        status_table.add_row("Files Processed", f"{processed_count}/{len(file_paths)}")
        if error_count > 0:
            status_table.add_row("Errors", f"{error_count}", style="red")
    except Exception as e:
        status_table.add_row("Processing", f"Error: {str(e)[:30]}")
    
    # Always add a minimum status even if empty
    if len(status_table.rows) == 0:
        status_table.add_row("Status", "Initializing...")

async def main():
    """Main application flow"""
    try:
        console.print(banner())
        check_tesseract_installation()
        check_ollama_status()  # Add Ollama status check
        
        config = load_config()
        current_directory = config.get("last_directory", os.getcwd())
        output_dir = config.get("output_directory", DEFAULT_OUTPUT_DIR)
        lang = config.get("last_language", "eng")
        
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
                model_name = config.get("selected_model", "yarn-llama2")
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
                
                results = await process_images(file_paths, lang)
                
                if not results:
                    console.print("[yellow]No results extracted.[/yellow]")
                    continue
                
                results_file = await save_results(results, output_dir)
                console.print(f"[green]OCR results saved to: {results_file}[/green]")
                
                console.print(display_result_summary(results))
                
                await analyze_with_ollama(results_file, output_dir)
            
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


