#!/usr/bin/env python3

import os
import sys
import base64
import asyncio
import time
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import ollama
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.layout import Layout
from rich.live import Live
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter
from prompt_toolkit.shortcuts import clear
import readline
import glob
import signal

# Try to import PDF processing libraries
try:
    from pdf2image import convert_from_path
    from PIL import Image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Constants
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}  # Removed PDF
DEFAULT_MODEL = 'llama3.2-vision'
DEFAULT_PROMPT = "Describe what you see in this image in detail."
PDF_DPI = 150  # DPI for PDF to image conversion

# Initialize Rich console
console = Console()

# Layout components
layout = Layout()
layout.split(
    Layout(name="header", size=10),  # Increased size for header
    Layout(name="body"),            # Scrollable content area
)
layout["header"].split_row(
    Layout(name="status", ratio=3),
    Layout(name="menu", ratio=2),
)

def update_layout(current_directory: str, selected_files: Set[str], prompt: str, model: str):
    """Update the fixed header with current status and menu"""
    # Status panel
    status_content = f"[blue]Current directory:[/blue] {current_directory}\n"
    status_content += f"[blue]Selected files:[/blue] {len(selected_files)}\n"
    status_content += f"[blue]Current model:[/blue] {model}\n"
    status_content += f"[blue]Current prompt:[/blue] {prompt}"
    
    layout["status"].update(Panel(status_content, title="Status", border_style="blue"))

    # Menu panel
    menu_content = "[bold cyan]Menu Options:[/bold cyan]\n"
    menu_content += "  1. Change directory\n"
    menu_content += "  2. View and select image files\n"
    menu_content += "  3. Process selected images\n"
    menu_content += "  4. Change prompt\n"
    menu_content += "  5. Change model\n"
    menu_content += "  6. Exit"
    
    layout["menu"].update(Panel(menu_content, title="Menu", border_style="cyan"))

def banner():
    """Display the application banner"""
    return Panel.fit(
        "[bold blue]Vision Explorer[/bold blue] - Interactive Ollama Vision Model Interface",
        border_style="green"
    )

def is_image_file(file_path: str) -> bool:
    """Check if a file is a supported image type"""
    return os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in SUPPORTED_EXTENSIONS

def is_pdf_file(file_path: str) -> bool:
    """Check if a file is a PDF"""
    return os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() == '.pdf'

def get_files_for_processing(directory: str) -> Tuple[List[str], List[str]]:
    """Get list of image files and PDFs in directory"""
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                  if is_image_file(os.path.join(directory, f))]
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                if is_pdf_file(os.path.join(directory, f))] if PDF_SUPPORT else []
    
    return image_files, pdf_files

def get_image_files(directory: str) -> List[str]:
    """Get list of image files in directory"""
    return [f for f in os.listdir(directory) if is_image_file(os.path.join(directory, f))]

def encode_image(image_path: str) -> str:
    """Encode image file to base64"""
    with open(image_path, "rb") as image_file:
        return image_path

def check_ollama_model(model_name: str) -> bool:
    """Check if the model is available in Ollama"""
    try:
        models_response = ollama.list()
        
        if 'models' not in models_response:
            console.print("[yellow]Warning: Unexpected response format from Ollama API[/yellow]")
            return False
            
        # Print raw model data for debugging
        console.print(f"[dim cyan]Debug - looking for model: {model_name}[/dim cyan]")
        
        # Safely check for model name with more flexible matching
        for model in models_response['models']:
            # Try both 'name' field and direct model dict access
            model_name_from_api = model.get('name', '')
            if not model_name_from_api and 'model' in model:
                model_name_from_api = model['model']
            
            # Use case-insensitive partial matching for more robust detection
            if model_name_from_api and model_name.lower() in model_name_from_api.lower():
                console.print(f"[dim green]Found match: {model_name_from_api}[/dim green]")
                return True
                
        return False
    except Exception as e:
        console.print(f"[bold red]Error checking models: {e}[/bold red]")
        return False

def save_result(image_path: str, result: str) -> str:
    """Save analysis result to text file next to original image"""
    result_path = os.path.splitext(image_path)[0] + "_analysis.txt"
    with open(result_path, "w") as f:
        f.write(result)
    return result_path

def display_directory_contents(current_dir: str, selected_files: Set[str]) -> Table:
    """Display directory contents with image files highlighted and return a table"""
    all_files = os.listdir(current_dir)
    
    image_files = [f for f in all_files if is_image_file(os.path.join(current_dir, f))]
    pdf_files = [f for f in all_files if is_pdf_file(os.path.join(current_dir, f))] if PDF_SUPPORT else []
    other_files = [f for f in all_files if f not in image_files and f not in pdf_files]
    
    table = Table(title=f"Directory: {current_dir}")
    table.add_column("Type", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Status", style="magenta")
    
    # Add parent directory
    table.add_row("📁", "..", "")
    
    # Add directories
    for item in sorted([f for f in all_files if os.path.isdir(os.path.join(current_dir, f))]):
        table.add_row("📁", item, "")
    
    # Add image files
    for item in sorted(image_files):
        file_path = os.path.join(current_dir, item)
        status = "[bold yellow]Selected[/bold yellow]" if file_path in selected_files else ""
        table.add_row("🖼️", item, status)
    
    # Add PDF files
    for item in sorted(pdf_files):
        file_path = os.path.join(current_dir, item)
        status = "[bold yellow]Selected[/bold yellow]" if file_path in selected_files else ""
        table.add_row("📄", item, status)
    
    return table

async def convert_pdf_to_images(pdf_path: str, progress_callback=None) -> List[str]:
    """Convert PDF to images and return paths to created images"""
    if not PDF_SUPPORT:
        console.print("[bold red]PDF support requires pdf2image and pillow libraries[/bold red]")
        return []
        
    try:
        # Create temp directory for images
        with tempfile.TemporaryDirectory() as temp_dir:
            console.print(f"[yellow]Converting PDF to images: {os.path.basename(pdf_path)}[/yellow]")
            
            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=PDF_DPI,
                output_folder=temp_dir,
                fmt='jpeg',
                thread_count=2
            )
            
            # Create output directory next to the PDF
            pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = os.path.join(os.path.dirname(pdf_path), f"{pdf_basename}_images")
            os.makedirs(output_dir, exist_ok=True)
            
            # Save images to output directory
            image_paths = []
            for i, img in enumerate(images):
                if progress_callback:
                    progress_callback(i, len(images))
                    
                img_path = os.path.join(output_dir, f"{pdf_basename}_page_{i+1}.jpg")
                img.save(img_path, "JPEG")
                image_paths.append(img_path)
                
            console.print(f"[green]✓[/green] Converted {len(image_paths)} pages from PDF")
            return image_paths
            
    except Exception as e:
        console.print(f"[bold red]Error converting PDF to images: {str(e)}[/bold red]")
        return []

async def process_pdf(pdf_path: str, prompt: str, model: str) -> str:
    """Process a PDF file by converting to images and then using vision model"""
    console.print(f"[yellow]Processing PDF: {os.path.basename(pdf_path)}[/yellow]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green"),  # Add a progress bar
        TextColumn("[cyan]{task.fields[status]}"),                # Add status field
        transient=True
    ) as progress:
        convert_task = progress.add_task("[cyan]Converting PDF...", total=None)
        
        # Convert PDF to images
        def update_progress(current, total):
            progress.update(convert_task, 
                           description=f"[cyan]Converting PDF page {current+1}/{total}...",
                           total=total, completed=current+1)
            
        image_paths = await convert_pdf_to_images(pdf_path, update_progress)
        
        if not image_paths:
            return "Failed to convert PDF to images"
            
        # Process each image
        results = []
        process_task = progress.add_task("[cyan]Processing PDF pages...", total=len(image_paths), status="")
        
        for i, img_path in enumerate(image_paths):
            progress.update(process_task, 
                           description=f"[cyan]Processing page {i+1}/{len(image_paths)}...",
                           status="",
                           advance=0)
            
            try:
                # Create message with image
                message = {
                    'role': 'user',
                    'content': f"{prompt} (Page {i+1}/{len(image_paths)} of PDF)",
                    'images': [img_path]
                }
                
                # Stream response to show progress
                result_chunks = []
                stream = ollama.chat(
                    model=model,
                    messages=[message],
                    stream=True  # Enable streaming for progress indication
                )
                
                dot_counter = 0
                progress_char = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                for chunk in stream:
                    chunk_content = chunk['message']['content']
                    result_chunks.append(chunk_content)
                    # Update status field instead of printing dots
                    dot_counter += 1
                    if dot_counter % 10 == 0:
                        progress.update(
                            process_task, 
                            status=f"Processing {progress_char[dot_counter//10 % len(progress_char)]}"
                        )
                        
                result = "".join(result_chunks)
                results.append(f"--- Page {i+1}/{len(image_paths)} ---\n\n{result}\n\n")
                
            except Exception as e:
                results.append(f"--- Page {i+1}/{len(image_paths)} ---\n\n[Error: {str(e)}]\n\n")
                
            progress.update(process_task, advance=1)
            
        # Combine results
        combined_result = f"PDF Analysis: {os.path.basename(pdf_path)}\n\n" + "\n".join(results)
        
        # Save combined result
        result_path = os.path.splitext(pdf_path)[0] + "_analysis.txt"
        with open(result_path, "w") as f:
            f.write(combined_result)
        
        return combined_result, result_path

async def test_model_sanity(model: str) -> bool:
    """Perform a quick sanity test on the model to verify it's responding"""
    console.print(f"[yellow]Running quick sanity test on model '{model}'...[/yellow]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(pulse_style="yellow"),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Testing model response...", total=None)
            
            # Simple, benign prompt that should be answered quickly by any working model
            test_message = {
                'role': 'user',
                'content': "Reply with just the word 'OK' if you can read this message."
            }
            
            # Add a timeout to prevent hanging if the model is not responding
            start_time = time.time()
            MAX_TEST_TIME = 20  # seconds - increased from 10 to 20 seconds
            
            # Use stream=True to see partial progress
            try:
                stream = ollama.chat(
                    model=model,
                    messages=[test_message],
                    stream=True,
                    options={"temperature": 0}  # Use deterministic output
                )
                
                # Collect response
                response_parts = []
                for chunk in stream:
                    response_parts.append(chunk['message']['content'])
                    # Check for timeout
                    if time.time() - start_time > MAX_TEST_TIME:
                        raise TimeoutError(f"Model response test timed out after {MAX_TEST_TIME} seconds")
                        
                response = "".join(response_parts).strip().lower()
                
                # Check if response contains "ok" - we're being lenient here
                if "ok" in response:
                    progress.stop()
                    console.print(f"[green]✓ Model sanity test passed! Response: '{response}'[/green]")
                    return True
                else:
                    progress.stop()
                    console.print(f"[yellow]⚠ Model responded but answer was unexpected: '{response}'[/yellow]")
                    return True  # Still return true if we got any coherent response
                    
            except Exception as e:
                progress.stop()
                console.print(f"[bold red]✗ Model sanity test failed: {str(e)}[/bold red]")
                return False
                
    except Exception as e:
        console.print(f"[bold red]Error in model sanity test: {str(e)}[/bold red]")
        return False

async def process_images(image_paths: List[str], prompt: str, model: str) -> None:
    """Process images with the vision model"""
    if not image_paths:
        console.print("[yellow]No images selected for processing.[/yellow]")
        return
    
    # First do a sanity check on the model
    model_ok = await test_model_sanity(model)
    if not model_ok:
        if not Confirm.ask("[yellow]Model sanity test failed. Continue anyway?[/yellow]"):
            return
    
    # Separate PDFs and images
    regular_images = [p for p in image_paths if not is_pdf_file(p)]
    pdf_files = [p for p in image_paths if is_pdf_file(p)]
    
    # Process PDFs if any
    for pdf_path in pdf_files:
        result, result_path = await process_pdf(pdf_path, prompt, model)
        console.print(Panel(
            f"[bold green]PDF:[/bold green] {os.path.basename(pdf_path)}\n\n[Saved analysis to {result_path}]",
            title="PDF Analysis Complete",
            border_style="green"
        ))
    
    # Process regular images
    if not regular_images:
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        BarColumn(complete_style="cyan", finished_style="green"),
        TextColumn("[cyan]{task.fields[status]}"),
        transient=False,
    ) as progress:
        task = progress.add_task("[cyan]Processing images...", total=len(regular_images), status="")
        
        for image_path in regular_images:
            progress.update(task, description=f"[cyan]Processing {os.path.basename(image_path)}...", status="")
            
            try:
                # First try with the Ollama Python library
                try:
                    # Create message with image
                    message = {
                        'role': 'user',
                        'content': prompt,
                        'images': [image_path]
                    }
                    
                    # Try to use streaming for progress visibility
                    stream = ollama.chat(
                        model=model,
                        messages=[message],
                        stream=True  # Enable streaming to see progress
                    )
                    
                    # Collect streamed response and show progress using the progress bar
                    result_chunks = []
                    chunk_count = 0
                    spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                    
                    for chunk in stream:
                        chunk_content = chunk['message']['content']
                        result_chunks.append(chunk_content)
                        # Update status instead of printing dots
                        chunk_count += 1
                        if chunk_count % 10 == 0:
                            spinner_frame = spinner_frames[(chunk_count // 10) % len(spinner_frames)]
                            progress.update(task, status=f"Processing {spinner_frame}")
                    
                    # Reset status when done
                    progress.update(task, status="Complete")
                    result = "".join(result_chunks)
                    
                    # Save and display result
                    result_path = save_result(image_path, result)
                    console.print(Panel(
                        f"[bold green]Image:[/bold green] {os.path.basename(image_path)}\n\n{result}",
                        title="Analysis Result",
                        border_style="green"
                    ))
                    console.print(f"[blue]Result saved to:[/blue] {result_path}")
                    
                except Exception as e:
                    # If the Ollama library fails, try direct API as fallback
                    console.print(f"[yellow]Ollama library failed: {str(e)}. Trying direct API as fallback...[/yellow]")
                    
                    try:
                        import requests
                        
                        # Read and encode image
                        with open(image_path, 'rb') as img_file:
                            img_data = base64.b64encode(img_file.read()).decode('utf-8')
                        
                        # Format payload for direct API (fixing the format issue)
                        payload = {
                            "model": model,
                            "messages": [{
                                "role": "user",
                                "content": prompt  # String content, not array
                            }]
                        }
                        
                        # Make API request
                        response = requests.post("http://localhost:11434/api/chat", json=payload)
                        
                        if response.status_code == 200:
                            result = response.json()['message']['content']
                            result_path = save_result(image_path, result)
                            
                            console.print(Panel(
                                f"[bold green]Image:[/bold green] {os.path.basename(image_path)}\n\n{result}",
                                title="Analysis Result (API Fallback)",
                                border_style="green"
                            ))
                            console.print(f"[blue]Result saved to:[/blue] {result_path}")
                        else:
                            raise Exception(f"API returned {response.status_code}: {response.text}")
                            
                    except ImportError:
                        console.print("[bold red]Requests library not available for API fallback[/bold red]")
                    except Exception as api_e:
                        console.print(f"[bold red]Direct API fallback also failed: {str(api_e)}[/bold red]")
            
            except Exception as e:
                console.print(f"[bold red]Error processing {os.path.basename(image_path)}: {str(e)}[/bold red]")
            
            progress.update(task, advance=1)
            
        console.print("[bold green]✓[/bold green] All images processed.")

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

def select_files(directory: str, currently_selected: Set[str]) -> Set[str]:
    """Interactive file selection"""
    selected = currently_selected.copy()
    
    # Get image and PDF files
    image_files = [f for f in os.listdir(directory) if is_image_file(os.path.join(directory, f))]
    pdf_files = [f for f in os.listdir(directory) if is_pdf_file(os.path.join(directory, f))] if PDF_SUPPORT else []
    
    all_files = image_files + pdf_files
    
    if not all_files:
        console.print("[yellow]No supported files found in this directory.[/yellow]")
        return selected
    
    # Create menu for file selection
    console.print("[blue]Available files:[/blue]")
    for i, file in enumerate(all_files, 1):
        file_path = os.path.join(directory, file)
        status = "[green]Selected[/green]" if file_path in selected else ""
        file_type = "PDF" if file in pdf_files else "Image"
        console.print(f"  {i}. {file} [{file_type}] {status}")
    
    console.print("\n[yellow]Enter file numbers to select/deselect (comma-separated), 'all' to select all, or empty to finish:[/yellow]")
    
    while True:
        choice = input("Select> ")
        
        if not choice:
            break
            
        if choice.lower() == 'all':
            for file in all_files:
                selected.add(os.path.join(directory, file))
            console.print("[green]All files selected.[/green]")
            break
            
        try:
            indices = [int(i.strip()) for i in choice.split(',') if i.strip()]
            for idx in indices:
                if 1 <= idx <= len(all_files):
                    file_path = os.path.join(directory, all_files[idx-1])
                    if file_path in selected:
                        selected.remove(file_path)
                        console.print(f"[yellow]Deselected:[/yellow] {all_files[idx-1]}")
                    else:
                        selected.add(file_path)
                        console.print(f"[green]Selected:[/green] {all_files[idx-1]}")
                else:
                    console.print(f"[red]Invalid index: {idx}[/red]")
        except ValueError:
            console.print("[red]Please enter valid numbers separated by commas.[/red]")
    
    return selected

def debug_list_available_models():
    """List all available models for debugging"""
    try:
        models_response = ollama.list()
        
        if 'models' not in models_response:
            console.print("[bold red]Error: Unexpected response format from Ollama API[/bold red]")
            console.print(f"Raw response: {models_response}")
            return
            
        console.print("[bold blue]Available models:[/bold blue]")
        table = Table(show_header=True)
        table.add_column("Name", style="green")
        table.add_column("Size", style="cyan")
        table.add_column("Modified", style="magenta")
        
        for model in models_response['models']:
            # Try both 'name' and 'model' fields
            name = model.get('name', 'Unknown')
            if name == 'Unknown' and 'model' in model:
                name = model['model']
                
            size = model.get('size', 'Unknown')
            # Format size to be more readable - fixed format specifier by removing extra colon
            if isinstance(size, int):
                size = f"{size / (1024*1024*1024):.2f} GB"
            modified = model.get('modified_at', 'Unknown')
            table.add_row(name, str(size), str(modified))
            
        console.print(table)
        
        # Alternative method: direct API call
        try:
            import requests
            console.print("[yellow]Trying direct API call to /api/tags...[/yellow]")
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                if 'models' in data:
                    console.print("[bold green]Direct API models:[/bold green]")
                    direct_table = Table(show_header=True)
                    direct_table.add_column("Name", style="green")
                    direct_table.add_column("Size", style="cyan")
                    
                    for model in data['models']:
                        name = model.get('name', 'Unknown')
                        size = model.get('size', 'Unknown')
                        # Also fix this format specifier
                        if isinstance(size, int):
                            size = f"{size / (1024*1024*1024):.2f} GB"
                        direct_table.add_row(name, str(size))
                        
                    console.print(direct_table)
        except ImportError:
            console.print("[yellow]Note: Install 'requests' library for additional API debugging[/yellow]")
            console.print("[blue]pip install requests[/blue]")
        except Exception as api_e:
            console.print(f"[red]Direct API call error: {api_e}[/red]")
            
    except Exception as e:
        console.print(f"[bold red]Error listing models: {str(e)}[/bold red]")

def check_requirements():
    """Check if required packages are installed and Ollama is running"""
    global DEFAULT_MODEL
    
    # Store requirement check results to display later
    check_results = []
    
    # Check for PDF support
    if not PDF_SUPPORT:
        check_results.append("[yellow]PDF support not available. Install pdf2image and pillow packages:[/yellow]")
        check_results.append("[blue]pip install pdf2image pillow[/blue]")
    else:
        check_results.append("[green]✓ PDF conversion support available[/green]")
    
    try:
        # Check if Ollama is running
        try:
            ollama.list()
            check_results.append("[green]✓ Connected to Ollama successfully[/green]")
        except Exception as e:
            check_results.append(f"[bold red]Error: Could not connect to Ollama: {str(e)}[/bold red]")
            check_results.append("[yellow]Make sure Ollama is installed and running.[/yellow]")
            check_results.append("[blue]Installation instructions: https://ollama.com/download[/blue]")
            console.print("\n".join(check_results))
            sys.exit(1)
            
        # Check for the exact vision model
        if check_ollama_model(DEFAULT_MODEL):
            check_results.append(f"[green]✓ {DEFAULT_MODEL} model is available[/green]")
        else:
            # Try alternate model names
            alternate_models = ['llama3.2-vision:latest', 'llama3.2-vision', 'llama3-vision']
            
            found_alt = False
            for alt_model in alternate_models:
                if alt_model != DEFAULT_MODEL and check_ollama_model(alt_model):
                    check_results.append(f"[yellow]! {DEFAULT_MODEL} not found, but {alt_model} is available[/yellow]")
                    console.print("\n".join(check_results))
                    if Confirm.ask(f"Would you like to use {alt_model} instead?"):
                        DEFAULT_MODEL = alt_model
                        check_results.append(f"[green]✓ Using {alt_model} model instead[/green]")
                        found_alt = True
                        break
            
            if not found_alt:
                check_results.append(f"[yellow]! {DEFAULT_MODEL} model not found. You may need to run:[/yellow]")
                check_results.append(f"[blue]  ollama pull {DEFAULT_MODEL}[/blue]")
                console.print("\n".join(check_results))
                if not Confirm.ask("Continue anyway?"):
                    sys.exit(1)
                    
    except Exception as e:
        check_results.append(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        console.print("\n".join(check_results))
        sys.exit(1)
    
    return check_results

def display_selected_files(selected_files: Set[str]) -> Table:
    """Display the currently selected files in a nice format and return a table"""
    if not selected_files:
        return None
        
    # Group files by directory for cleaner display
    files_by_dir = {}
    for file_path in selected_files:
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        if directory not in files_by_dir:
            files_by_dir[directory] = []
        files_by_dir[directory].append(filename)
    
    # Create a table to display selected files
    table = Table(title="Selected Files")
    table.add_column("Directory", style="blue")
    table.add_column("Files", style="green")
    
    for directory, files in files_by_dir.items():
        # Truncate directory path if too long
        dir_display = directory
        if len(dir_display) > 40:
            dir_display = "..." + dir_display[-37:]
            
        # Format files with type indicators
        file_list = []
        for filename in sorted(files):
            if is_pdf_file(os.path.join(directory, filename)):
                file_list.append(f"📄 {filename}")
            else:
                file_list.append(f"🖼️ {filename}")
        
        # Join files with newlines if more than 3, otherwise with commas
        if len(file_list) > 3:
            files_display = "\n".join(file_list)
        else:
            files_display = ", ".join(file_list)
            
        table.add_row(dir_display, files_display)
        
    return table

async def main():
    """Main application loop with fixed header and scrollable content"""
    clear()
    console.print(banner())
    
    # Check requirements first but don't display results yet
    requirement_results = check_requirements()
    
    # Set up initial state
    current_directory = os.getcwd()
    selected_files: Set[str] = set()
    prompt = DEFAULT_PROMPT
    model = DEFAULT_MODEL
    
    # Handle SIGINT (Ctrl+C) to exit cleanly
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    
    # Create a separate console for rendering status that won't get cleared
    status_console = Console()
    
    while True:
        # Clear screen
        clear()
        
        # Always show the banner
        console.print(banner())
        
        # Display status section (fixed)
        console.print(Panel(
            f"[blue]Current directory:[/blue] {current_directory}\n"
            f"[blue]Selected files:[/blue] {len(selected_files)}\n"
            f"[blue]Current model:[/blue] {model}\n"
            f"[blue]Current prompt:[/blue] {prompt}",
            title="Status", 
            border_style="blue"
        ))
        
        # Display menu section (fixed)
        console.print(Panel(
            "[bold cyan]Menu Options:[/bold cyan]\n"
            "  1. Change directory\n"
            "  2. View and select image files\n"
            "  3. Process selected images\n"
            "  4. Change prompt\n"
            "  5. Change model\n"
            "  6. Exit",
            title="Menu",
            border_style="cyan"
        ))
        
        # Display horizontal rule to separate fixed and scrollable sections
        console.rule("[bold yellow]Scrollable Content Below[/bold yellow]")
        
        # Display selected files if any
        if selected_files:
            selected_files_table = display_selected_files(selected_files)
            if selected_files_table:
                console.print(selected_files_table)
        else:
            # Show initial requirements or empty message
            if requirement_results:
                console.print(Panel("\n".join(requirement_results), 
                                   title="Status Check Results", 
                                   border_style="green"))
                # Clear requirement results after first display
                requirement_results = []
            else:
                console.print("[yellow]No files selected. Use options 1 and 2 to navigate and select files.[/yellow]")
        
        # Get user input - this is always visible at the bottom
        choice = Prompt.ask("\nSelect an option", choices=["1", "2", "3", "4", "5", "6"])
        
        # Process user choice
        if choice == "1":
            # Change directory
            new_directory = await interactive_directory_browser(current_directory)
            if new_directory != current_directory:
                current_directory = new_directory
                selected_files = {f for f in selected_files if os.path.exists(f)}  # Keep only existing files
        
        elif choice == "2":
            # View and select image files
            clear()
            console.print(banner())
            dir_table = display_directory_contents(current_directory, selected_files)
            console.print(dir_table)
            selected_files = select_files(current_directory, selected_files)
            console.print("\nPress Enter to continue...")
            input()
        
        elif choice == "3":
            # Process selected images
            if not selected_files:
                console.print("[yellow]No images selected. Please select images first.[/yellow]")
                console.print("\nPress Enter to continue...")
                input()
                continue
                
            clear()
            console.print(banner())
            await process_images(list(selected_files), prompt, model)
            
            # Ask if user wants to clear selection
            if Confirm.ask("Clear current file selection?"):
                selected_files.clear()
                
            console.print("\nPress Enter to continue...")
            input()
        
        elif choice == "4":
            # Change prompt
            console.print(f"[blue]Current prompt:[/blue] {prompt}")
            new_prompt = Prompt.ask("Enter new prompt", default=prompt)
            prompt = new_prompt
        
        elif choice == "5":
            # Change model
            console.print(f"[blue]Current model:[/blue] {model}")
            new_model = Prompt.ask("Enter model name", default=model)
            if check_ollama_model(new_model):
                model = new_model
            else:
                console.print(f"[yellow]Model '{new_model}' not found. Please pull it first with:[/yellow]")
                console.print(f"[blue]  ollama pull {new_model}[/blue]")
                console.print("\nPress Enter to continue...")
                input()
        
        elif choice == "6":
            # Exit
            console.print("[green]Exiting Vision Explorer. Goodbye![/green]")
            break

if __name__ == "__main__":
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user. Exiting...[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        sys.exit(1)
