#!/usr/bin/env python3

import os
import sys
import base64
import asyncio
import time
import re
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
import ollama
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

# Try to import PDF processing libraries
try:
    from pdf2image import convert_from_path
    from PIL import Image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Constants
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.JPG', '.PNG'}
PDF_EXTENSIONS = {'.pdf', '.PDF'}
DEFAULT_MODEL = 'llama3.2-vision'
DEFAULT_PROMPT = "Look at this image and identify any dates of correspondence or events mentioned. Return ONLY the date in MM.DD.YY format (e.g., 11.05.24). If multiple dates are present, return the most prominent one. If no date is found, respond with 'NO_DATE_FOUND'."
PDF_DPI = 150

# Initialize Rich console
console = Console()

# Layout components
layout = Layout()
layout.split(
    Layout(name="header", size=10),
    Layout(name="body"),
)

def banner():
    """Display the application banner"""
    return Panel.fit(
        "[bold blue]Date Extractor[/bold blue] - Extract dates from images using Ollama Vision Model",
        border_style="green"
    )

def is_image_file(file_path: str) -> bool:
    """Check if a file is a supported image type"""
    return os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in map(str.lower, SUPPORTED_EXTENSIONS)

def is_pdf_file(file_path: str) -> bool:
    """Check if a file is a PDF"""
    return os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in map(str.lower, PDF_EXTENSIONS)

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

async def extract_date_from_image(image_path: str, prompt: str, model: str) -> str:
    """Extract date from image using Ollama vision model"""
    try:
        # Create message with image
        message = {
            'role': 'user',
            'content': prompt,
            'images': [image_path]
        }

        # Get response from model
        response = ollama.chat(
            model=model,
            messages=[message]
        )

        if response and 'message' in response and 'content' in response['message']:
            return response['message']['content'].strip()
        else:
            return "NO_DATE_FOUND"
    except Exception as e:
        console.print(f"[bold red]Error extracting date from {os.path.basename(image_path)}: {str(e)}[/bold red]")
        return "ERROR_EXTRACTING_DATE"

def validate_and_format_date(date_text: str) -> str:
    """Validate and format the extracted date"""
    # First, clean up the response to get just the date
    date_text = date_text.strip()
    
    # Check if no date was found
    if "NO_DATE_FOUND" in date_text or not date_text:
        return None
        
    # Try to extract date with regex if the model returned extra text
    date_patterns = [
        # MM.DD.YY format
        r'(\d{1,2}\.\d{1,2}\.\d{2,4})',
        # MM/DD/YY format
        r'(\d{1,2}/\d{1,2}/\d{2,4})',
        # MM-DD-YY format
        r'(\d{1,2}-\d{1,2}-\d{2,4})',
        # "Month DD, YYYY" format
        r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})[\s,]+(\d{2,4})'
    ]

    for pattern in date_patterns:
        matches = re.search(pattern, date_text)
        if matches:
            # Handle the "Month DD, YYYY" format
            if len(matches.groups()) > 1 and matches.group(1) in ["January", "February", "March", "April", "May", "June", 
                                                                  "July", "August", "September", "October", "November", "December"]:
                month_names = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                               "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
                month = month_names[matches.group(1)]
                day = int(matches.group(2))
                year = int(matches.group(3))
                if year < 100:  # Handle 2-digit year
                    year = year + 2000 if year < 50 else year + 1900
                return f"{month:02d}.{day:02d}.{str(year)[2:]}"
            else:
                # For other formats, extract and standardize to MM.DD.YY
                date_str = matches.group(1)
                # Replace separators with dots
                date_str = re.sub(r'[/\-]', '.', date_str)
                # Ensure YY format for year
                date_parts = date_str.split('.')
                if len(date_parts) == 3 and len(date_parts[2]) == 4:  # YYYY format
                    date_parts[2] = date_parts[2][2:]  # Convert to YY
                return '.'.join(date_parts)
                
    # If no pattern matched but we have something that looks like a date (has numbers)
    if re.search(r'\d', date_text):
        # Just return the first line as a fallback
        return date_text.split("\n")[0][:20]  # Limit to 20 chars
                
    return None

def rename_file_with_date(file_path: str, date_str: str) -> str:
    """Rename file to include date at the beginning"""
    if not date_str:
        return file_path  # No change if no date found
        
    directory = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    
    # Check if filename already starts with a date pattern
    if re.match(r'^\d{1,2}\.\d{1,2}\.\d{2}', filename):
        console.print(f"[yellow]File already has date prefix: {filename}[/yellow]")
        return file_path
    
    # Get file extension
    name_parts = os.path.splitext(filename)
    base_name = name_parts[0]
    extension = name_parts[1]
    
    # Create new filename with date prefix
    new_filename = f"{date_str}.{base_name}{extension}"
    new_path = os.path.join(directory, new_filename)
    
    # If file with this name already exists, add a counter
    counter = 1
    while os.path.exists(new_path):
        new_filename = f"{date_str}.{base_name} ({counter}){extension}"
        new_path = os.path.join(directory, new_filename)
        counter += 1
    
    # Rename the file
    try:
        os.rename(file_path, new_path)
        return new_path
    except Exception as e:
        console.print(f"[bold red]Error renaming file {filename}: {str(e)}[/bold red]")
        return file_path

async def process_directory(directory: str, prompt: str, model: str, recursive: bool = False) -> None:
    """Process all images in a directory, extracting dates and renaming files"""
    # Get all files in directory
    files = []
    if recursive:
        for root, _, filenames in os.walk(directory):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                if is_image_file(file_path) or is_pdf_file(file_path):
                    files.append(file_path)
    else:
        files = [os.path.join(directory, f) for f in os.listdir(directory) 
                 if is_image_file(os.path.join(directory, f)) or is_pdf_file(os.path.join(directory, f))]
    
    if not files:
        console.print("[yellow]No supported files found in directory.[/yellow]")
        return
    
    # Create a log file
    log_path = os.path.join(directory, "date_extraction_log.txt")
    with open(log_path, "w") as log_file:
        log_file.write(f"Date Extraction Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Directory: {directory}\n")
        log_file.write(f"Model: {model}\n")
        log_file.write(f"Prompt: {prompt}\n\n")
        log_file.write("=" * 80 + "\n\n")
    
    # Process files with progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(complete_style="cyan", finished_style="green"),
        TextColumn("[cyan]{task.fields[status]}"),
        TimeElapsedColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        main_task = progress.add_task(f"[cyan]Processing {len(files)} files...", total=len(files), status="")
        
        for file_path in files:
            filename = os.path.basename(file_path)
            progress.update(main_task, description=f"[cyan]Processing {filename}...", status="")
            
            try:
                # Handle PDFs differently
                if is_pdf_file(file_path):
                    # Log PDF processing start
                    with open(log_path, "a") as log_file:
                        log_file.write(f"PDF: {filename}\n")
                    
                    # Convert PDF to images
                    def update_progress(current, total):
                        progress.update(main_task, status=f"Converting PDF page {current+1}/{total}")
                        
                    pdf_images = await convert_pdf_to_images(file_path, update_progress)
                    
                    # Process each PDF image
                    for i, img_path in enumerate(pdf_images):
                        img_filename = os.path.basename(img_path)
                        progress.update(main_task, status=f"Extracting date from PDF page {i+1}/{len(pdf_images)}")
                        
                        # Extract date from image
                        date_text = await extract_date_from_image(img_path, prompt, model)
                        
                        # Validate and format date
                        formatted_date = validate_and_format_date(date_text)
                        
                        # Rename file with date
                        if formatted_date:
                            new_path = rename_file_with_date(img_path, formatted_date)
                            
                            # Log result
                            with open(log_path, "a") as log_file:
                                log_file.write(f"  Page {i+1}: {img_filename} -> {os.path.basename(new_path)}\n")
                                log_file.write(f"  Extracted text: {date_text}\n")
                                log_file.write(f"  Formatted date: {formatted_date}\n\n")
                        else:
                            # Log no date found
                            with open(log_path, "a") as log_file:
                                log_file.write(f"  Page {i+1}: {img_filename} -> No date found\n")
                                log_file.write(f"  Model response: {date_text}\n\n")
                else:
                    # Regular image file
                    progress.update(main_task, status="Extracting date")
                    
                    # Extract date from image
                    date_text = await extract_date_from_image(file_path, prompt, model)
                    
                    # Validate and format date
                    formatted_date = validate_and_format_date(date_text)
                    
                    # Rename file with date
                    original_path = file_path
                    if formatted_date:
                        new_path = rename_file_with_date(file_path, formatted_date)
                        
                        # Log result
                        with open(log_path, "a") as log_file:
                            log_file.write(f"Image: {filename} -> {os.path.basename(new_path)}\n")
                            log_file.write(f"Extracted text: {date_text}\n")
                            log_file.write(f"Formatted date: {formatted_date}\n\n")
                    else:
                        # Log no date found
                        with open(log_path, "a") as log_file:
                            log_file.write(f"Image: {filename} -> No date found\n")
                            log_file.write(f"Model response: {date_text}\n\n")
                
            except Exception as e:
                console.print(f"[bold red]Error processing {filename}: {str(e)}[/bold red]")
                # Log error
                with open(log_path, "a") as log_file:
                    log_file.write(f"ERROR on {filename}: {str(e)}\n\n")
            
            progress.update(main_task, advance=1)
    
    console.print(f"[bold green]✓[/bold green] Processing complete! Log saved to: {log_path}")

def check_ollama_model(model_name: str) -> bool:
    """Check if the model is available in Ollama"""
    try:
        models_response = ollama.list()
        
        if 'models' not in models_response:
            console.print("[yellow]Warning: Unexpected response format from Ollama API[/yellow]")
            return False
            
        # Check for model name
        for model in models_response['models']:
            model_name_from_api = model.get('name', '')
            if not model_name_from_api and 'model' in model:
                model_name_from_api = model['model']
            
            if model_name_from_api and model_name.lower() in model_name_from_api.lower():
                return True
                
        return False
    except Exception as e:
        console.print(f"[bold red]Error checking models: {e}[/bold red]")
        return False

def check_requirements():
    """Check if required packages are installed and Ollama is running"""
    global DEFAULT_MODEL
    
    # Check for PDF support
    if not PDF_SUPPORT:
        console.print("[yellow]PDF support not available. Install pdf2image and pillow packages:[/yellow]")
        console.print("[blue]pip install pdf2image pillow[/blue]")
        
    try:
        # Check if Ollama is running
        try:
            ollama.list()
            console.print("[green]✓ Connected to Ollama successfully[/green]")
        except Exception as e:
            console.print(f"[bold red]Error: Could not connect to Ollama: {str(e)}[/bold red]")
            console.print("[yellow]Make sure Ollama is installed and running.[/yellow]")
            console.print("[blue]Installation instructions: https://ollama.com/download[/blue]")
            sys.exit(1)
            
        # Check for the vision model
        if check_ollama_model(DEFAULT_MODEL):
            console.print(f"[green]✓ {DEFAULT_MODEL} model is available[/green]")
        else:
            # Try alternate model names
            alternate_models = ['llava', 'bakllava', 'llama3-vision']
            
            found_alt = False
            for alt_model in alternate_models:
                if check_ollama_model(alt_model):
                    console.print(f"[yellow]! {DEFAULT_MODEL} not found, but {alt_model} is available[/yellow]")
                    if Confirm.ask(f"Would you like to use {alt_model} instead?"):
                        DEFAULT_MODEL = alt_model
                        console.print(f"[green]✓ Using {alt_model} model instead[/green]")
                        found_alt = True
                        break
            
            if not found_alt:
                console.print(f"[yellow]! {DEFAULT_MODEL} model not found. You may need to run:[/yellow]")
                console.print(f"[blue]  ollama pull {DEFAULT_MODEL}[/blue]")
                if not Confirm.ask("Continue anyway?"):
                    sys.exit(1)
                    
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        sys.exit(1)

async def display_file_summary(directory: str):
    """Display summary of files in directory"""
    image_files = [f for f in os.listdir(directory) if is_image_file(os.path.join(directory, f))]
    pdf_files = [f for f in os.listdir(directory) if is_pdf_file(os.path.join(directory, f))]
    
    if not image_files and not pdf_files:
        console.print("[yellow]No supported files found in this directory.[/yellow]")
        return
    
    table = Table(title=f"Files in {directory}")
    table.add_column("Type", style="cyan")
    table.add_column("Count", style="green")
    table.add_column("Extensions", style="blue")
    
    # Group image files by extension
    image_exts = {}
    for f in image_files:
        ext = os.path.splitext(f)[1].lower()
        image_exts[ext] = image_exts.get(ext, 0) + 1
    
    # Add image files row
    table.add_row(
        "Images", 
        str(len(image_files)),
        ", ".join(f"{ext} ({count})" for ext, count in sorted(image_exts.items()))
    )
    
    # Group PDF files by extension
    pdf_exts = {}
    for f in pdf_files:
        ext = os.path.splitext(f)[1].lower()
        pdf_exts[ext] = pdf_exts.get(ext, 0) + 1
    
    # Add PDF files row
    if pdf_files:
        table.add_row(
            "PDFs", 
            str(len(pdf_files)),
            ", ".join(f"{ext} ({count})" for ext, count in sorted(pdf_exts.items()))
        )
    
    console.print(table)
    
    # Display sample of file names
    sample_size = min(5, len(image_files) + len(pdf_files))
    sample_files = (image_files + pdf_files)[:sample_size]
    
    console.print(f"\n[blue]Sample of {sample_size} files:[/blue]")
    for f in sample_files:
        if is_image_file(os.path.join(directory, f)):
            console.print(f"  [cyan]🖼️ {f}[/cyan]")
        else:
            console.print(f"  [magenta]📄 {f}[/magenta]")

async def main():
    """Main application flow"""
    clear()
    console.print(banner())
    
    check_requirements()
    
    # Set up initial state
    current_directory = os.getcwd()
    prompt = DEFAULT_PROMPT
    model = DEFAULT_MODEL
    recursive = False
    
    # Get directory to process
    console.print("\n[bold]Step 1: Select directory containing images[/bold]")
    current_directory = await interactive_directory_browser(current_directory)
    
    # Display summary of files
    clear()
    console.print(banner())
    console.print(f"\n[bold]Directory selected:[/bold] {current_directory}")
    await display_file_summary(current_directory)
    
    # Confirm recursive processing
    if Confirm.ask("\nProcess subdirectories recursively?", default=False):
        recursive = True
        console.print("[yellow]Will process all images in subdirectories.[/yellow]")
    
    # Customize prompt if needed
    console.print(f"\n[bold]Step 2: Customize date extraction prompt[/bold]")
    console.print(f"[blue]Current prompt:[/blue] {prompt}")
    
    if Confirm.ask("Would you like to customize the prompt?", default=False):
        new_prompt = Prompt.ask("Enter new prompt", default=prompt)
        prompt = new_prompt
    
    # Customize model if needed
    console.print(f"\n[bold]Step 3: Select vision model[/bold]")
    console.print(f"[blue]Current model:[/blue] {model}")
    
    if Confirm.ask("Would you like to use a different model?", default=False):
        new_model = Prompt.ask("Enter model name", default=model)
        if check_ollama_model(new_model):
            model = new_model
        else:
            console.print(f"[yellow]Model '{new_model}' not found. Please pull it first with:[/yellow]")
            console.print(f"[blue]  ollama pull {new_model}[/blue]")
            if Confirm.ask("Continue with original model?", default=True):
                pass
            else:
                return
    
    # Final confirmation
    clear()
    console.print(banner())
    console.print(f"[bold]Ready to process images:[/bold]")
    console.print(f"[blue]Directory:[/blue] {current_directory} {'(including subdirectories)' if recursive else ''}")
    console.print(f"[blue]Model:[/blue] {model}")
    console.print(f"[blue]Prompt:[/blue] {prompt}")
    
    if not Confirm.ask("\nProceed with date extraction and file renaming?", default=True):
        console.print("[yellow]Operation cancelled.[/yellow]")
        return
    
    # Process directory
    await process_directory(current_directory, prompt, model, recursive)
    
    # Ask if user wants to open directory
    if Confirm.ask("\nWould you like to open the directory to view results?", default=True):
        try:
            if sys.platform == 'darwin':  # macOS
                os.system(f'open "{current_directory}"')
            elif sys.platform == 'win32':  # Windows
                os.system(f'explorer "{current_directory}"')
            elif sys.platform == 'linux':  # Linux
                os.system(f'xdg-open "{current_directory}"')
        except Exception as e:
            console.print(f"[yellow]Could not open directory: {str(e)}[/yellow]")

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
