#!/usr/bin/env python3

import os
import sys
import json
import time  # Added missing import for time module
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import cv2
import numpy as np
import pytesseract
from PIL import Image
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.layout import Layout
from rich.live import Live
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.shortcuts import clear

# Constants
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.JPG', '.PNG', '.JPEG', '.TIFF', '.pdf', '.PDF'}
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/ocr_results")

# Add these constants for environment variables
ENV_LAST_DIRECTORY = "OCR_EXTRACTOR_LAST_DIR"
ENV_OUTPUT_DIRECTORY = "OCR_EXTRACTOR_OUTPUT_DIR"

# Add new constants for better AI prompts
ANALYSIS_PROMPT = """
You are an expert in organizing and analyzing text communications.

I have OCR results from multiple screenshots of text messages or conversations.
Please analyze these texts and:

1. Create a strict chronological timeline of the conversation
2. Identify all participants and their roles
3. Connect related messages and threads
4. Summarize the key points or decisions made

Format your analysis as a beautiful, well-structured markdown document with:
- A clear title and introduction 
- Messages organized by date and time
- Clear indication of senders and recipients
- Visual separation of different conversation threads
- Highlights of important information or decisions

Use markdown formatting effectively to create an easy-to-read document.
"""

SIMPLE_PROMPT = """
Please organize these OCR results from text message screenshots into a chronological timeline.
For each message, include:
- Sender
- Date and time
- Message content
- Any relevant context

Format your response as markdown with clear headings and separation between messages.
"""

# Update constants for config persistence
CONFIG_DIR = os.path.expanduser("~/.ocr_extractor")
CONFIG_FILE = os.path.join(CONFIG_DIR, "config.json")

def load_config() -> Dict[str, Any]:
    """Load configuration from file"""
    default_config = {
        "last_directory": os.getcwd(),
        "output_directory": DEFAULT_OUTPUT_DIR,
        "selected_files_count": "0",
        "last_language": "eng"
    }
    
    # Create config directory if it doesn't exist
    try:
        os.makedirs(CONFIG_DIR, exist_ok=True)
        
        # If config file exists, load it
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            # Create default config file
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

# Try to detect Tesseract installation
try:
    if sys.platform == 'win32':
        # Check common Windows installation paths
        tesseract_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        for path in tesseract_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
except Exception:
    pass  # Will handle Tesseract errors during runtime

# Initialize Rich console
console = Console()

def banner():
    """Display the application banner"""
    return Panel.fit(
        "[bold blue]OCR Text Extractor[/bold blue] - Extract text from images to reconstruct message timelines",
        border_style="green"
    )

def is_supported_file(file_path: str) -> bool:
    """Check if a file is a supported image type"""
    return os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in map(str.lower, SUPPORTED_EXTENSIONS)

async def interactive_directory_browser(start_dir: str = os.getcwd()) -> str:
    """Interactive directory browser with autocomplete"""
    console.clear()
    console.print(banner())
    
    session = PromptSession()
    completer = PathCompleter(
        only_directories=True,
        expanduser=True,
    )
    
    current_dir = os.path.abspath(start_dir)
    
    while True:
        try:
            console.print(f"[blue]Current directory:[/blue] {current_dir}")
            console.print("[yellow]Type directory path or '..' to go up, or press Enter to select this directory[/yellow]")
            
            user_input = await session.prompt_async("Directory> ", completer=completer)
            
            if not user_input:
                return current_dir
            
            # Handle special case for parent directory
            if user_input == "..":
                current_dir = os.path.dirname(current_dir)
                continue
                
            # Handle relative or absolute path
            if os.path.isabs(user_input):
                path = user_input
            else:
                path = os.path.join(current_dir, user_input)
                
            # Normalize the path (resolve .. and symlinks)
            path = os.path.normpath(os.path.expanduser(path))
            
            if os.path.isdir(path):
                current_dir = path
            else:
                console.print("[bold red]Not a valid directory. Please try again.[/bold red]")
                
        except KeyboardInterrupt:
            raise
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")

def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess the image to improve OCR results"""
    # Read the image using OpenCV
    image = cv2.imread(image_path)
    
    if image is None:
        # Try with PIL if OpenCV fails
        pil_image = Image.open(image_path)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        if image is None:
            raise ValueError(f"Could not open image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get black text on white background
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    return denoised

def extract_text_from_image(image_path: str, lang: str = "eng") -> str:
    """Extract text from image using OCR"""
    try:
        # Special handling for PDF files
        if image_path.lower().endswith('.pdf'):
            from pdf2image import convert_from_path
            pages = convert_from_path(image_path, 300)
            text_results = []
            
            for page_num, page in enumerate(pages):
                text = pytesseract.image_to_string(page, lang=lang)
                text_results.append(text)
                
            return "\n\n".join(text_results)
        else:
            # Regular image processing
            # First try with preprocessing
            try:
                preprocessed = preprocess_image(image_path)
                text = pytesseract.image_to_string(preprocessed, lang=lang)
            except Exception:
                # If preprocessing fails, try direct OCR
                img = Image.open(image_path)
                text = pytesseract.image_to_string(img, lang=lang)
            
            return text
    except Exception as e:
        console.print(f"[bold red]Error processing {image_path}: {str(e)}[/bold red]")
        return f"ERROR: {str(e)}"

def get_image_metadata(image_path: str) -> Dict[str, Any]:
    """Extract metadata from image"""
    try:
        # Get basic file metadata
        file_stats = os.stat(image_path)
        file_name = os.path.basename(image_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        # Try to get image dimensions and more detailed metadata
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                format_name = img.format
                mode = img.mode
                
                # Try to get EXIF data
                exif_data = {}
                if hasattr(img, '_getexif') and img._getexif():
                    exif = {
                        pytesseract.pytesseract.TesseractError: v
                        for k, v in img._getexif().items()
                        if k in pytesseract.pytesseract.TesseractError
                    }
                    exif_data = exif
        except Exception:
            width, height = None, None
            format_name = file_ext.lstrip('.')
            mode = None
            exif_data = {}
        
        return {
            "filename": file_name,
            "path": image_path,
            "size_bytes": file_stats.st_size,
            "modified_time": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
            "created_time": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "width": width,
            "height": height,
            "format": format_name,
            "mode": mode,
            "exif": exif_data
        }
    except Exception as e:
        return {
            "filename": os.path.basename(image_path),
            "path": image_path,
            "error": str(e)
        }

async def select_files_for_analysis(directory: str) -> List[str]:
    """Allow user to select which files to include in analysis"""
    all_files = [f for f in os.listdir(directory) if is_supported_file(os.path.join(directory, f))]
    
    if not all_files:
        console.print("[yellow]No supported image files found in this directory.[/yellow]")
        return []
    
    # Sort files by name for easier selection
    all_files.sort()
    
    # Create table of files
    table = Table(title=f"Images in {directory}")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Filename", style="green")
    table.add_column("Size", style="magenta")
    
    for i, filename in enumerate(all_files, 1):
        file_path = os.path.join(directory, filename)
        file_size = os.path.getsize(file_path)
        size_str = f"{file_size / 1024:.1f} KB" if file_size < 1024 * 1024 else f"{file_size / (1024 * 1024):.1f} MB"
        table.add_row(str(i), filename, size_str)
    
    console.print(table)
    console.print(f"[blue]Found {len(all_files)} supported image files in this directory[/blue]")
    
    # Options for selection
    console.print("\n[bold]Selection options:[/bold]")
    console.print("1. Select all files")
    console.print("2. Select files by number (comma-separated)")
    console.print("3. Select files by pattern match")
    
    choice = Prompt.ask("Choose an option", choices=["1", "2", "3"], default="1")
    
    selected_files = []
    
    if choice == "1":
        selected_files = [os.path.join(directory, f) for f in all_files]
        console.print(f"[green]✓ Selected all {len(selected_files)} images[/green]")
    
    elif choice == "2":
        while True:
            selection = Prompt.ask("Enter file numbers (comma-separated, e.g. 1,3,5-9)")
            try:
                selected_indices = []
                parts = selection.split(',')
                
                for part in parts:
                    part = part.strip()
                    if '-' in part:
                        # Handle ranges (e.g., 5-9)
                        start, end = map(int, part.split('-'))
                        selected_indices.extend(range(start, end + 1))
                    else:
                        # Handle single numbers
                        selected_indices.append(int(part))
                
                # Validate indices
                valid_indices = [idx for idx in selected_indices if 1 <= idx <= len(all_files)]
                
                if not valid_indices:
                    console.print("[red]No valid file numbers selected.[/red]")
                    continue
                
                selected_files = [os.path.join(directory, all_files[i-1]) for i in valid_indices]
                console.print(f"[green]✓ Selected {len(selected_files)} images[/green]")
                break
            
            except ValueError:
                console.print("[red]Invalid input format. Please try again.[/red]")
    
    elif choice == "3":
        pattern = Prompt.ask("Enter filename pattern (e.g. 'screenshot' or 'msg')")
        matched_files = [f for f in all_files if pattern.lower() in f.lower()]
        
        if not matched_files:
            console.print(f"[yellow]No files matched the pattern '{pattern}'.[/yellow]")
            return await select_files_for_analysis(directory)
        
        console.print(f"[green]Found {len(matched_files)} files matching '{pattern}':[/green]")
        for f in matched_files[:10]:  # Show first 10 matches
            console.print(f"  - {f}")
        
        if len(matched_files) > 10:
            console.print(f"  ... and {len(matched_files) - 10} more")
        
        if Confirm.ask(f"Select these {len(matched_files)} files?", default=True):
            selected_files = [os.path.join(directory, f) for f in matched_files]
            console.print(f"[green]✓ Selected {len(selected_files)} images[/green]")
        else:
            return await select_files_for_analysis(directory)
    
    # Save selection count to show in the main menu - using config instead of env vars
    if selected_files:
        config = load_config()
        config["selected_files_count"] = str(len(selected_files))
        save_config(config)
        
    return selected_files

async def process_images(file_paths: List[str], lang: str = "eng") -> List[Dict[str, Any]]:
    """Process images and extract OCR text"""
    if not file_paths:
        console.print("[yellow]No files selected for processing.[/yellow]")
        return []
    
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green"),
        TextColumn("[cyan]{task.fields[status]}"),
        TimeElapsedColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        main_task = progress.add_task(f"[cyan]Processing {len(file_paths)} images...", total=len(file_paths), status="")
        
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            progress.update(main_task, description=f"[cyan]Processing {filename}...", status="")
            
            try:
                # Extract OCR text
                ocr_text = extract_text_from_image(file_path, lang)
                
                # Get image metadata
                metadata = get_image_metadata(file_path)
                
                # Create result object
                result = {
                    "filename": filename,
                    "file_path": file_path,
                    "metadata": metadata,
                    "text": ocr_text,
                    "processed_at": datetime.now().isoformat()
                }
                
                results.append(result)
                progress.update(main_task, advance=1, status="Complete")
                
            except Exception as e:
                console.print(f"[bold red]Error processing {filename}: {str(e)}[/bold red]")
                results.append({
                    "filename": filename,
                    "file_path": file_path,
                    "error": str(e),
                    "processed_at": datetime.now().isoformat()
                })
                progress.update(main_task, advance=1, status="Error")
    
    return results

async def save_results(results: List[Dict[str, Any]], output_dir: str) -> str:
    """Save OCR results to JSON file"""
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Create output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"ocr_results_{timestamp}.json")
    
    # Create outer JSON structure
    output_data = {
        "created_at": datetime.now().isoformat(),
        "file_count": len(results),
        "results": results
    }
    
    # Write to file
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
        
        if "error" in result:
            table.add_row(filename, "ERROR", result["error"])
            continue
            
        text = result.get("text", "")
        text_len = str(len(text))
        
        # Truncate preview for display
        preview = text.replace("\n", " ")[:50] + "..." if len(text) > 50 else text.replace("\n", " ")
        
        table.add_row(filename, text_len, preview)
    
    return table

def check_tesseract_installation() -> bool:
    """Check if Tesseract OCR is properly installed"""
    try:
        # Try to get tesseract version
        version = pytesseract.get_tesseract_version()
        console.print(f"[green]✓ Tesseract OCR detected (version {version})[/green]")
        return True
    except Exception as e:
        console.print("[bold red]Tesseract OCR not found or not properly configured.[/bold red]")
        console.print("[yellow]Please install Tesseract OCR and ensure it's in your PATH.[/yellow]")
        
        # Show installation instructions based on platform
        if sys.platform == 'win32':
            console.print("[blue]Windows installation:[/blue]")
            console.print("1. Download installer from https://github.com/UB-Mannheim/tesseract/wiki")
            console.print("2. Install and add to PATH")
            console.print("3. Set pytesseract.pytesseract.tesseract_cmd to installation path")
        elif sys.platform == 'darwin':
            console.print("[blue]macOS installation:[/blue]")
            console.print("brew install tesseract")
        else:
            console.print("[blue]Linux installation:[/blue]")
            console.print("sudo apt-get install tesseract-ocr")
        
        return False

def get_ollama_models() -> List[str]:
    """Get a list of available Ollama models"""
    try:
        import ollama
        try:
            # Direct method to get models - more reliable
            try:
                # Try first with the list API
                response = ollama.list()
                
                # Debug the response
                console.print(f"[dim]Debug - Ollama response: {type(response)}[/dim]")
                
                # Properly handle different response formats from different Ollama versions
                if isinstance(response, dict) and "models" in response:
                    models = response["models"]
                    model_names = []
                    
                    # Extract model names with better debugging
                    for model in models:
                        console.print(f"[dim]Debug - Model entry: {model}[/dim]")
                        if isinstance(model, dict):
                            # Different versions of ollama API return model names differently
                            if "name" in model:
                                model_names.append(model["name"])
                            elif "model" in model:
                                model_names.append(model["model"])
                    
                    return model_names
                else:
                    # Alternative approach for newer Ollama versions
                    console.print("[yellow]Using alternative approach to get models[/yellow]")
                    # Some Ollama versions might return a list directly
                    if isinstance(response, list):
                        return [m.get("name", m.get("model", str(m))) for m in response if isinstance(m, dict)]
                    else:
                        # Fallback to running CLI command for models
                        import subprocess
                        result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                        if result.returncode == 0:
                            # Parse the output to extract model names
                            lines = result.stdout.strip().split('\n')[1:]  # Skip header line
                            return [line.split()[0] for line in lines if line.strip()]
                        
                        return ["llama2", "mistral", "gemma:7b"]  # Default fallback models
            except Exception as e:
                console.print(f"[yellow]Error with primary model detection: {str(e)}[/yellow]")
                # Fallback to running CLI command
                import subprocess
                try:
                    result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                    if result.returncode == 0:
                        # Parse the output to extract model names
                        lines = result.stdout.strip().split('\n')[1:]  # Skip header line
                        return [line.split()[0] for line in lines if line.strip()]
                except Exception:
                    pass
                
                # Last resort - suggest common models
                return ["llama2", "mistral", "gemma:7b"]
                
        except Exception as e:
            console.print(f"[yellow]Error listing Ollama models: {str(e)}[/yellow]")
            return ["llama2", "mistral", "phi"]  # Return default options
            
    except ImportError:
        console.print("[red]Ollama package not installed. Try: pip install ollama[/red]")
        return []

def test_ollama_connection() -> bool:
    """Test if Ollama server is running and accessible"""
    try:
        import ollama
        try:
            # A simple API call to test the connection
            ollama.list()
            return True
        except Exception as e:
            console.print(f"[yellow]Could not connect to Ollama server: {str(e)}[/yellow]")
            console.print("[blue]Make sure Ollama is installed and running. Visit https://ollama.com/download[/blue]")
            return False
    except ImportError:
        console.print("[red]Ollama package not installed. Try: pip install ollama[/red]")
        return False

def pull_ollama_model(model_name: str) -> bool:
    """Pull an Ollama model if it doesn't exist"""
    try:
        import ollama
        
        console.print(f"[cyan]Pulling model: {model_name}[/cyan]")
        console.print("[yellow]This may take several minutes depending on your internet connection and the model size.[/yellow]")
        
        try:
            # Using subprocess for better visibility of the downloading process
            import subprocess
            import sys
            
            if sys.platform == "win32":
                # Windows needs shell=True for command line tools
                process = subprocess.Popen(
                    f"ollama pull {model_name}", 
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            else:
                # Unix-like systems
                process = subprocess.Popen(
                    ["ollama", "pull", model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
            
            # Show pull progress
            for line in process.stdout:
                console.print(f"[dim]{line.strip()}[/dim]")
            
            process.wait()
            if process.returncode == 0:
                console.print(f"[green]✓ Successfully pulled model: {model_name}[/green]")
                return True
            else:
                console.print(f"[red]Failed to pull model: {model_name}[/red]")
                return False
                
        except Exception as e:
            # Fallback to using the Python API if subprocess fails
            console.print(f"[yellow]Using ollama Python API to pull model: {str(e)}[/yellow]")
            ollama.pull(model_name)
            console.print(f"[green]✓ Successfully pulled model: {model_name}[/green]")
            return True
            
    except ImportError:
        console.print("[red]Ollama package not installed. Try: pip install ollama[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Error pulling model: {str(e)}[/red]")
        return False

# Function to copy images to output directory
def copy_images_to_output(file_paths: List[str], output_dir: str) -> Dict[str, str]:
    """Copy image files to output directory and return mapping of original to new paths"""
    image_map = {}
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    console.print("[cyan]Copying images to output directory...[/cyan]")
    
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        # Create a safe filename with timestamp to avoid collisions
        base, ext = os.path.splitext(filename)
        safe_filename = f"{base}_{int(time.time())}_{hash(file_path) % 10000:04d}{ext}"
        output_path = os.path.join(images_dir, safe_filename)
        
        try:
            # Copy the file
            import shutil
            shutil.copy2(file_path, output_path)
            # Store mapping from original path to new relative path for markdown
            image_map[file_path] = os.path.join("images", safe_filename)
        except Exception as e:
            console.print(f"[yellow]Error copying {filename}: {str(e)}[/yellow]")
    
    return image_map

# Enhanced function to create beautiful markdown output
async def create_markdown_summary(data: List[Dict[str, Any]], results_file: str, image_map: Dict[str, str] = None) -> str:
    """Create a beautiful markdown summary of the OCR results"""
    # Determine output path based on results file
    output_dir = os.path.dirname(results_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"ocr_summary_{timestamp}.md")
    
    # Sort data by date if possible, handling unknown dates
    try:
        # First, try to parse dates and create a sortable list
        dated_items = []
        undated_items = []
        
        for item in data:
            filename = item.get("filename", "unknown")
            text = item.get("text", "")
            # Try to find anything that looks like a date in the text
            if "date" not in item or item["date"] == "unknown":
                # If no date field, put in undated
                undated_items.append(item)
            else:
                dated_items.append(item)
        
        # Sort dated items
        dated_items.sort(key=lambda x: x.get("date", "9999-99-99"))
        
        # Combine lists with dated items first
        sorted_data = dated_items + undated_items
    except Exception:
        # If sorting fails, just use the original data
        sorted_data = data
    
    # Create markdown content
    markdown = [
        "# OCR Text Extraction Summary",
        f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
        "",
        "## Contents",
        "1. [Overview](#overview)",
        "2. [Extracted Messages](#extracted-messages)",
        "3. [Analysis](#analysis)",
        "",
        "## Overview",
        f"This document contains OCR text extracted from {len(data)} images.",
        "",
        "## Extracted Messages",
        ""
    ]
    
    # Add each message with image reference if available
    for i, item in enumerate(sorted_data, 1):
        filename = item.get("filename", "unknown")
        file_path = item.get("file_path", "")
        text = item.get("text", "").strip()
        
        # Try to get date information
        date_info = item.get("date", "Unknown date")
        sender_info = item.get("sender", "Unknown sender")
        
        # Create section header
        markdown.append(f"### Message {i}: {sender_info} ({date_info})")
        
        # Add image reference if we have the mapping
        if image_map and file_path in image_map:
            rel_path = image_map[file_path]
            markdown.append(f"[View original image]({rel_path})")
        
        # Add filename
        markdown.append(f"*Source: {filename}*")
        
        # Add the OCR text in a code block for clarity
        markdown.append("")
        markdown.append("```")
        markdown.append(text)
        markdown.append("```")
        markdown.append("")
    
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown))
    
    return output_path

# Enhanced send_to_model function with better console handling
async def send_to_model(results_file: str, prompt_style: str = "detailed"):
    """Send the OCR results to Ollama for analysis and generate beautiful markdown output"""
    try:
        import ollama
        import time
        
        # First check if Ollama server is running
        try:
            # Simple test to check server connection
            ollama.list()
            console.print("[green]✓ Connected to Ollama server[/green]")
        except Exception as e:
            console.print(f"[bold red]Error connecting to Ollama server: {str(e)}[/bold red]")
            console.print("[yellow]Please make sure Ollama is installed and running.[/yellow]")
            console.print("[blue]Installation instructions: https://ollama.com/download[/blue]")
            return False
        
        # Load the OCR results
        try:
            # Add debug info about results file path
            console.print(f"[dim]Debug - Loading OCR results from: {results_file}[/dim]")
            
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_results = data.get("results", [])
                if not all_results:
                    console.print("[yellow]Warning: No results found in the results file.[/yellow]")
        except Exception as e:
            console.print(f"[bold red]Error loading results file: {str(e)}[/bold red]")
            return False  # Return False to indicate failure
        
        # Ask user if they want to copy images to output directory
        include_images = Confirm.ask(
            "Would you like to copy images to the output directory for references?", 
            default=True
        )
        
        image_map = {}
        if include_images:
            try:
                # Get list of file paths from the results
                file_paths = [item.get("file_path") for item in all_results if "file_path" in item]
                if not file_paths:
                    console.print("[yellow]Warning: No valid file paths found in results.[/yellow]")
                
                # Copy images and get mapping
                output_dir = os.path.dirname(results_file)
                image_map = copy_images_to_output(file_paths, output_dir)
            except Exception as e:
                console.print(f"[yellow]Error copying images: {str(e)}[/yellow]")
                # Continue even if image copying fails
        
        # First, create a markdown summary of the OCR results
        try:
            markdown_path = await create_markdown_summary(all_results, results_file, image_map)
            console.print(f"[green]✓ Created initial markdown summary: {markdown_path}[/green]")
        except Exception as e:
            console.print(f"[yellow]Error creating markdown summary: {str(e)}[/yellow]")
            markdown_path = None
            
        # Create prompt for the model
        if prompt_style == "simple":
            base_prompt = SIMPLE_PROMPT
        else:
            base_prompt = ANALYSIS_PROMPT
            
        # Select prompt detail level
        prompt_options = ["Basic", "Detailed", "Custom"]
        console.print("\n[bold]Select prompt detail level:[/bold]")
        for i, option in enumerate(prompt_options, 1):
            console.print(f"{i}. {option}")
            
        prompt_choice = Prompt.ask("Choose option", choices=["1", "2", "3"], default="2")
        
        if prompt_choice == "1":
            prompt = SIMPLE_PROMPT
        elif prompt_choice == "2":
            prompt = ANALYSIS_PROMPT
        else:
            # Show the detailed prompt and allow customization
            console.print("[blue]Current prompt template:[/blue]")
            console.print(Panel(ANALYSIS_PROMPT, title="Analysis Prompt", border_style="blue"))
            prompt = Prompt.ask("Enter custom prompt", default=ANALYSIS_PROMPT)
        
        # Prepare simple version of data for the prompt
        simplified_data = []
        for result in all_results:
            simplified_entry = {
                "filename": result.get("filename", "unknown"),
                "text": result.get("text", ""),
            }
            
            # Try to extract date information if available
            if "metadata" in result:
                metadata = result.get("metadata", {})
                if "created_time" in metadata:
                    simplified_entry["timestamp"] = metadata["created_time"]
                elif "modified_time" in metadata:
                    simplified_entry["timestamp"] = metadata["modified_time"]
            
            simplified_data.append(simplified_entry)
        
        # Add data to prompt
        prompt += "\n\nHere's the OCR data:\n" + json.dumps(simplified_data, indent=2)
        
        # Try to get models or suggest defaults
        model_names = get_ollama_models()
        
        if not model_names:
            console.print("[yellow]Failed to detect Ollama models automatically.[/yellow]")
            
            # Offer to try common models directly
            suggested_models = ["llama2", "mistral", "phi", "llama3", "gemma:7b"]
            console.print("[bold]Common models you could try:[/bold]")
            for i, model in enumerate(suggested_models, 1):
                console.print(f"{i}. {model}")
            
            model_choice = Prompt.ask(
                "Choose a model number or type a custom model name",
                default="1"
            )
            
            try:
                idx = int(model_choice) - 1
                if 0 <= idx < len(suggested_models):
                    model_name = suggested_models[idx]
                else:
                    model_name = model_choice
            except ValueError:
                model_name = model_choice
        else:
            # If we successfully got models, display them
            console.print("\n[bold]Available Ollama models:[/bold]")
            table = Table(title="Ollama Models", show_header=True)
            table.add_column("#", style="cyan", justify="right")
            table.add_column("Model", style="green")
            
            for i, name in enumerate(model_names, 1):
                table.add_row(str(i), name)
            
            console.print(table)
            
            # Ask user to select a model
            model_input = Prompt.ask(
                "Choose model number or enter model name", 
                default="1" if model_names else "llama2"
            )
            
            # Determine selected model
            try:
                idx = int(model_input) - 1
                if 0 <= idx < len(model_names):
                    model_name = model_names[idx]
                else:
                    model_name = model_input
            except ValueError:
                model_name = model_input
        
        # Ask if the user wants to pull the model if it's not in the list
        if model_names and model_name not in model_names:
            console.print(f"[yellow]Model '{model_name}' is not in the detected model list.[/yellow]")
            console.print("[yellow]You can still try to use it if it's installed.[/yellow]")
            
            if Confirm.ask(f"Would you like to pull '{model_name}' first?", default=True):
                pull_ollama_model(model_name)
        
        # Now use the model to analyze
        console.print(f"[cyan]Using model: {model_name}[/cyan]", flush=True)
        
        # Instead of using Progress, use direct console messages to avoid buffering issues
        console.print("[cyan]Starting analysis with Ollama...[/cyan]", flush=True)
        console.print("[cyan]This may take a few minutes depending on the model and amount of text...[/cyan]", flush=True)
        
        # Record start time to show elapsed time
        start_time = time.time()
        
        try:
            # Call Ollama with appropriate parameters
            console.print("[cyan]Sending request to Ollama model...[/cyan]", flush=True)
            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.2,
                    "num_predict": 4096
                }
            )
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            console.print(f"[green]✓ Analysis completed in {elapsed_time:.2f} seconds[/green]", flush=True)
            
            # More detailed debugging for response format
            console.print(f"[dim]Debug - Response type: {type(response)}[/dim]", flush=True)
            
            # Handle different Ollama API versions and response formats
            analysis = None
            
            # Method 1: Standard dict with message->content structure
            if isinstance(response, dict) and "message" in response and "content" in response["message"]:
                analysis = response["message"]["content"]
                console.print("[green]✓ Successfully extracted content from response[/green]", flush=True)
            
            # Method 2: ChatResponse object with message attribute (newer Ollama versions)
            elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                analysis = response.message.content
                console.print("[green]✓ Successfully extracted content from response.message.content[/green]", flush=True)
            
            # Method 3: Response object with content attribute
            elif hasattr(response, 'content'):
                analysis = response.content
                console.print("[green]✓ Successfully extracted content from response.content[/green]", flush=True)
            
            # Method 4: Look for content directly in response for older Ollama versions
            elif isinstance(response, dict) and "content" in response:
                analysis = response["content"]
                console.print("[green]✓ Successfully extracted content from response['content'][/green]", flush=True)
            
            # If we found content through any method, proceed with saving to file
            if analysis:
                # Save analysis as markdown with reference to images
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_dir = os.path.dirname(results_file)
                output_path = os.path.join(output_dir, f"timeline_analysis_{timestamp}.md")
                
                console.print(f"[cyan]Writing analysis to: {output_path}[/cyan]", flush=True)
                
                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Create a complete markdown document
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        # Write a header
                        f.write("# OCR Text Analysis and Timeline\n\n")
                        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')} using {model_name}*\n\n")
                        
                        # Write the model's analysis
                        f.write(analysis)
                        
                        # Add links to original source material at the bottom
                        f.write("\n\n## Source Material\n\n")
                        f.write("- [Raw OCR Data](" + os.path.basename(results_file) + ")\n")
                        if markdown_path:
                            f.write("- [Basic OCR Summary](" + os.path.basename(markdown_path) + ")\n\n")
                        
                        # Add a list of all images used with thumbnails if images were copied
                        if include_images and image_map:
                            f.write("\n## Original Images\n\n")
                            f.write("| Filename | Preview |\n")
                            f.write("|----------|--------|\n")
                            
                            for original_path, copied_path in image_map.items():
                                filename = os.path.basename(original_path)
                                f.write(f"| [{filename}]({copied_path}) | ![]({copied_path}) |\n")
                                
                    console.print(f"[green]✓ Analysis saved to: {output_path}[/green]", flush=True)
                    
                    # Verify the file was created
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        console.print(f"[green]✓ Confirmed file exists: {output_path} ({file_size} bytes)[/green]", flush=True)
                    else:
                        console.print(f"[red]! File was not created: {output_path}[/red]", flush=True)
                        
                except Exception as file_error:
                    console.print(f"[bold red]Error writing markdown file: {str(file_error)}[/bold red]", flush=True)
                    import traceback
                    console.print(traceback.format_exc(), flush=True)
                    return False
                
                # Show preview of analysis
                console.print("\n[bold]Analysis Preview:[/bold]", flush=True)
                preview = analysis[:2000] + "..." if len(analysis) > 2000 else analysis
                console.print(Panel(preview, title=f"Model: {model_name}", border_style="green", width=100), flush=True)
                
                # Ask if user wants to view full analysis
                if Confirm.ask("Open analysis in text editor?", default=True):
                    try:
                        if sys.platform == 'darwin':  # macOS
                            os.system(f'open "{output_path}"')
                        elif sys.platform == 'win32':  # Windows
                            os.system(f'start "" "{output_path}"')
                        else:  # Linux
                            os.system(f'xdg-open "{output_path}"')
                    except Exception as e:
                        console.print(f"[yellow]Could not open file: {str(e)}[/yellow]", flush=True)
                        
                return True  # Return True to indicate success
                
            # Rest of the error handling code with flush=True added to all console.print calls
            # ...existing code but add flush=True to all console.print calls...
            
        except Exception as e:
            console.print(f"[bold red]Error during analysis: {str(e)}[/bold red]", flush=True)
            console.print("[yellow]Try a different model or check your Ollama installation.[/yellow]", flush=True)
            return False
    
    except ImportError:
        console.print("[red]Ollama package not installed. Try: pip install ollama[/red]", flush=True)
        return False
    except Exception as e:
        console.print(f"[bold red]Unexpected error during model analysis: {str(e)}[/bold red]", flush=True)
        import traceback
        console.print(traceback.format_exc(), flush=True)
        return False

# Fix the process_all_files function to also use flush=True
async def process_all_files(directory: str, output_dir: str, lang: str = "eng"):
    """Process all images in a directory and run the complete workflow"""
    console.print("\n[bold]Quick Process: Processing all image files in directory[/bold]", flush=True)
    
    try:
        # Step 1: Find all supported images
        console.print("[cyan]Looking for supported image files...[/cyan]", flush=True)
        file_paths = [os.path.join(directory, f) for f in os.listdir(directory) 
                    if is_supported_file(os.path.join(directory, f))]
        
        if not file_paths:
            console.print("[yellow]No supported image files found in this directory.[/yellow]", flush=True)
            return False
            
        # Step 2: Process images
        console.print(f"[cyan]Found {len(file_paths)} images. Processing...[/cyan]", flush=True)
        results = await process_images(file_paths, lang)
        
        if not results:
            console.print("[yellow]No results extracted.[/yellow]", flush=True)
            return False
        
        # Step 3: Save results
        console.print("[cyan]Saving OCR results to file...[/cyan]", flush=True)
        results_file = await save_results(results, output_dir)
        console.print(f"[green]OCR results saved to: {results_file}[/green]", flush=True)
        
        # Verify results file was created
        if not os.path.exists(results_file):
            console.print(f"[red]! Results file does not exist at: {results_file}[/red]", flush=True)
            return False
            
        # Display a summary of the extracted data
        console.print(display_result_summary(results), flush=True)
        
        # Step 4: Generate analysis - use detailed prompt by default
        console.print("\n[bold]Generating AI analysis...[/bold]", flush=True)
        
        # We need to ensure this completes, so wait for it
        success = await send_to_model(results_file, "detailed")
        
        if success:
            console.print("[green]✓ AI analysis completed successfully![/green]", flush=True)
            return True
        else:
            console.print("[yellow]AI analysis did not complete successfully.[/yellow]", flush=True)
            return False
            
    except Exception as e:
        console.print(f"[bold red]Error in process_all_files: {str(e)}[/bold red]", flush=True)
        import traceback
        console.print(traceback.format_exc(), flush=True)
        return False

def check_dependencies():
    """Check if required packages are installed and configured"""
    # Check for Tesseract
    if not check_tesseract_installation():
        if not Confirm.ask("Continue without Tesseract OCR?", default=False):
            sys.exit(1)
    
    # Check for PDF support if needed
    try:
        import pdf2image
        console.print("[green]✓ PDF support available[/green]")
    except ImportError:
        console.print("[yellow]PDF support not available. Install pdf2image for PDF processing.[/yellow]")
        console.print("pip install pdf2image poppler-utils")
    
    # Check for Ollama support
    try:
        import ollama
        console.print("[green]✓ Ollama package installed[/green]")
        
        # Test connection to Ollama server
        if test_ollama_connection():
            console.print("[green]✓ Connected to Ollama server successfully[/green]")
            
            # List available models
            models = get_ollama_models()
            if models:
                console.print(f"[green]✓ Found {len(models)} Ollama models[/green]")
            else:
                console.print("[yellow]No Ollama models found. You can pull models later.[/yellow]")
        else:
            console.print("[yellow]Ollama server not running or not accessible.[/yellow]")
            console.print("[blue]You can still use OCR functionality without Ollama.[/blue]")
    except ImportError:
        console.print("[yellow]Ollama package not installed. Install with: pip install ollama[/yellow]")
        console.print("[blue]You can still use OCR functionality, but AI analysis will be unavailable.[/blue]")

# Update main function with reorganized menu items
async def main():
    """Main application flow"""
    try:
        clear()
        console.print(banner())
        
        # Check dependencies
        check_dependencies()
        
        # Load configuration from file
        config = load_config()
        
        # Set up initial state from config
        current_directory = config.get("last_directory", os.getcwd())
        lang = config.get("last_language", "eng")
        output_dir = config.get("output_directory", DEFAULT_OUTPUT_DIR)
        selected_files_count = config.get("selected_files_count", "0")
        
        # Validate loaded paths
        if not os.path.isdir(current_directory):
            console.print(f"[yellow]Saved directory '{current_directory}' not found. Using current directory.[/yellow]")
            current_directory = os.getcwd()
            
        if not os.path.isdir(output_dir):
            console.print(f"[yellow]Saved output directory '{output_dir}' not found. Using default.[/yellow]")
            output_dir = DEFAULT_OUTPUT_DIR
        
        # Main application loop
        while True:
            clear()
            console.print(banner())
            
            # Reorganized menu with improved logical flow
            console.print("\n[bold cyan]Menu Options:[/bold cyan]")
            # Directory operations first
            console.print("  1. Select directory containing images")
            console.print("  2. Set output directory")
            # Configuration settings next
            console.print("  3. Select images for OCR processing")
            console.print("  4. Change OCR language")
            # Execution operations
            console.print("  5. Process images with OCR")
            console.print("  6. Send to AI for analysis")
            console.print("  7. Quick Process All (Do everything at once)")
            # Exit last
            console.print("  0. Exit")
            
            # Current status display with more information
            console.print(f"\n[blue]Current directory:[/blue] {current_directory}")
            console.print(f"[blue]Current OCR language:[/blue] {lang}")
            console.print(f"[blue]Output directory:[/blue] {output_dir}")
            
            # Show selected files count if available
            if 'file_paths' in locals() and file_paths:
                console.print(f"[blue]Selected files:[/blue] {len(file_paths)}")
            elif selected_files_count != "0":
                console.print(f"[blue]Selected files:[/blue] {selected_files_count} (from previous selection)")
            
            # Get user choice - updated for new menu numbering
            choice = Prompt.ask("\nChoose an option", choices=["0", "1", "2", "3", "4", "5", "6", "7"], default="1")
            
            if choice == "1":
                # Select directory with images (formerly choice 1, unchanged)
                console.print("\n[bold]Select directory containing images[/bold]")
                try:
                    prev_dir = current_directory
                    current_directory = await interactive_directory_browser(current_directory)
                    
                    # Save to config if changed
                    if prev_dir != current_directory:
                        config["last_directory"] = current_directory
                        # Clear selected files count if directory changed
                        config["selected_files_count"] = "0"
                        save_config(config)
                        selected_files_count = "0"
                        
                        if 'file_paths' in locals():
                            del file_paths  # Clear selected files when directory changes
                        
                except Exception as e:
                    console.print(f"[yellow]Error with directory browser: {str(e)}[/yellow]")
            
            elif choice == "2":
                # Set output directory (formerly choice 5)
                console.print("\n[bold]Set Output Directory[/bold]")
                console.print(f"Current output directory: {output_dir}")
                
                if Confirm.ask("Change output directory?", default=True):
                    prev_dir = output_dir
                    new_dir = await interactive_directory_browser(output_dir)
                    output_dir = new_dir
                    
                    # Save to config if changed
                    if prev_dir != output_dir:
                        config["output_directory"] = output_dir
                        save_config(config)
                        
                    console.print(f"[green]Output directory set to: {output_dir}[/green]")
            
            elif choice == "3":
                # Select files for analysis (formerly choice 2)
                console.print("\n[bold]Select images for OCR processing[/bold]")
                file_paths = await select_files_for_analysis(current_directory)
                
                if not file_paths:
                    console.print("[yellow]No files selected.[/yellow]")
                    continue
                
                # Update selected_files_count from config
                config = load_config()
                selected_files_count = config.get("selected_files_count", "0")
            
            elif choice == "4":
                # Change OCR language (formerly choice 3)
                console.print("\n[bold]Change OCR Language[/bold]")
                console.print("[blue]Common language codes:[/blue]")
                console.print("eng = English")
                console.print("fra = French")
                console.print("deu = German")
                console.print("spa = Spanish")
                console.print("chi_sim = Chinese Simplified")
                console.print("jpn = Japanese")
                console.print("rus = Russian")
                console.print("ara = Arabic")
                console.print("Use '+' to combine multiple languages (e.g., 'eng+spa')")
                
                new_lang = Prompt.ask("Enter OCR language code", default=lang)
                lang = new_lang
                
                # Save language to config
                config["last_language"] = lang
                save_config(config)
                
                console.print(f"[green]OCR language set to: {lang}[/green]")
            
            elif choice == "5":
                # Process images with OCR (formerly choice 4)
                console.print("\n[bold]Process images with OCR[/bold]")
                if not 'file_paths' in locals() or not file_paths:
                    console.print("[yellow]No files selected. Please select files first (option 3).[/yellow]")
                    continue
                    
                results = await process_images(file_paths, lang)
                
                if not results:
                    console.print("[yellow]No results extracted.[/yellow]")
                    continue
                
                # Display summary of extracted data
                console.print(display_result_summary(results))
                
                # Save results
                results_file = await save_results(results, output_dir)
                console.print(f"[green]OCR results saved to: {results_file}[/green]")
                
                # Ask if user wants to see full text of a specific result
                if Confirm.ask("View full text of a specific result?", default=False):
                    idx = int(Prompt.ask("Enter result number", default="1")) - 1
                    if 0 <= idx < len(results):
                        result = results[idx]
                        filename = result.get("filename", "unknown")
                        text = result.get("text", "No text extracted")
                        console.print(Panel(text, title=f"OCR Text: {filename}", border_style="blue"))
                    else:
                        console.print("[yellow]Invalid result number.[/yellow]")
            
            elif choice == "6":
                # Send to model for analysis
                console.print("\n[bold]Send to AI for Analysis[/bold]", flush=True)
                
                # Critical fix: Check if results_file exists in local scope
                if 'results_file' not in locals() or not os.path.exists(results_file):
                    if 'results' in locals() and results:
                        # If we have results but no saved file, save them first
                        results_file = await save_results(results, output_dir)
                        console.print(f"[green]OCR results saved to: {results_file}[/green]", flush=True)
                    else:
                        console.print("[yellow]No OCR results available. Process images first (option 5).[/yellow]", flush=True)
                        continue
                
                # Now we're sure results_file exists
                success = await send_to_model(results_file)
                if success:
                    console.print("[green]✓ AI analysis completed successfully![/green]", flush=True)
                else:
                    console.print("[yellow]AI analysis did not complete successfully.[/yellow]", flush=True)
            
            elif choice == "7":
                # Quick Process All - new option with improved handling (formerly choice 8)
                console.print("\n[bold]Quick Process All[/bold]")
                console.print("[yellow]This will process all supported images in the current directory and generate analysis.[/yellow]")
                
                if Confirm.ask("Proceed with quick process?", default=True):
                    try:
                        # Ensure output directory exists
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Run the all-in-one process and wait for it to complete
                        success = await process_all_files(current_directory, output_dir, lang)
                        
                        if success:
                            console.print("[green]✓ Quick process completed successfully![/green]")
                        else:
                            console.print("[yellow]Quick process did not complete successfully.[/yellow]")
                            
                        # Add a prompt to continue after the process completes
                        console.print("\nPress Enter to return to main menu...")
                        input()
                    except Exception as e:
                        console.print(f"[bold red]Error during quick process: {str(e)}[/bold red]")
                        import traceback
                        console.print(traceback.format_exc())
            
            elif choice == "0":
                # Exit (formerly choice 7) - now using standard menu convention with 0 for exit
                console.print("[cyan]Exiting application.[/cyan]")
                
                # Update config with latest settings
                config["last_directory"] = current_directory
                config["output_directory"] = output_dir
                config["last_language"] = lang
                save_config(config)
                
                break
                
            # Add a small pause between menu displays
            if choice != "0":  # Don't pause after exit
                if choice != "7":  # Option 7 (Quick Process) already has its own pause
                    console.print("\nPress Enter to continue...")
                    input()
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user. Exiting...[/yellow]")
        
        # Still save config even on keyboard interrupt
        try:
            config = load_config()
            if 'current_directory' in locals():
                config["last_directory"] = current_directory
            if 'output_dir' in locals():
                config["output_directory"] = output_dir
            if 'lang' in locals():
                config["last_language"] = lang
            save_config(config)
        except Exception:
            pass
            
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    # Ensure proper event loop handling
    if sys.platform == 'win32':
        import asyncio
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    else:
        import asyncio
        asyncio.run(main())
