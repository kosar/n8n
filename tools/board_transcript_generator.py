#!/usr/bin/env python3

import os
import sys
import base64
import asyncio
import time
import re
import tempfile
import json
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

# Add OpenAI client for DeepSeek API
try:
    from openai import OpenAI
    DEEPSEEK_SUPPORT = True
except ImportError:
    DEEPSEEK_SUPPORT = False

# Try to import PDF processing libraries
try:
    from pdf2image import convert_from_path
    from PIL import Image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Constants
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.JPG', '.PNG', '.JPEG', '.screenshot'}
DEFAULT_MODEL = 'llama3.2-vision'
DEFAULT_DEEPSEEK_MODEL = 'deepseek-vision'
DEFAULT_DEEPSEEK_API_URL = 'https://api.deepseek.com'
DEFAULT_PROVIDER = 'ollama'  # 'ollama' or 'deepseek'

DEFAULT_PROMPT = """
Study this screenshot of communication between 11 West building board members about an incident where the live-in superintendent's dog bit a worker.

Extract the following information:
1. Date and time of message (be as precise as possible)
2. Sender name
3. Recipients (if visible)
4. Complete message content
5. Communication platform (email, text message, chat app, etc.)
6. Any contextual details that help place this in the timeline

Format your response as JSON:
{
  "date": "YYYY-MM-DD HH:MM",
  "sender": "Full Name",
  "recipients": ["Person 1", "Person 2"],
  "platform": "email/text/etc",
  "content": "Full message content",
  "context": "Additional context"
}

If the date format in the image is different, convert it to YYYY-MM-DD format. If time isn't available, just use the date. If you're uncertain about any information, include your best guess but mark it with '(uncertain)'.
"""

ANALYSIS_PROMPT = """
You are analyzing a collection of screenshots related to an incident where the live-in superintendent's dog bit a worker at 11 West building.

I've provided you with JSON data extracted from multiple screenshots. This data contains messages from different dates, times, senders, and platforms.

Your task:
1. Organize all communications in strict chronological order
2. If the exact time isn't available for some messages, use contextual clues to place them appropriately
3. Create a coherent narrative transcript that shows the conversation flow
4. Identify any inconsistencies or gaps in the timeline
5. Note where participants are referencing previous communications
6. Include all relevant details about the incident

Format your response as a markdown document with:
- A title section summarizing the incident
- A chronological transcript with clear timestamps, senders, and recipients
- Clear indications of which platform each communication occurred on
- Brief notes on how you determined the chronological order when timestamps were unclear

This document should read like a complete timeline of the events surrounding the dog bite incident.
"""

# Initialize Rich console
console = Console()

def banner():
    """Display the application banner"""
    return Panel.fit(
        "[bold blue]Board Transcript Generator[/bold blue] - Create chronological transcripts from communication screenshots",
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

def encode_image_to_base64(image_path: str) -> str:
    """Encode an image file to base64 string for API calls"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def extract_message_data(image_path: str, prompt: str, model: str, provider: str, api_key: str = None, api_url: str = None) -> Dict[str, Any]:
    """Extract message data from screenshot using selected vision model provider"""
    try:
        if provider == 'ollama':
            # Create message with image for Ollama
            message = {
                'role': 'user',
                'content': prompt,
                'images': [image_path]
            }

            # Get response from Ollama
            response = ollama.chat(
                model=model,
                messages=[message],
                options={"temperature": 0.2}  # Lower temperature for more deterministic outputs
            )

            if response and 'message' in response and 'content' in response['message']:
                raw_response = response['message']['content'].strip()
            else:
                raise Exception("Invalid response from Ollama")
                
        elif provider == 'deepseek':
            if not api_key:
                raise ValueError("DeepSeek API key is required")
                
            # Initialize OpenAI client with DeepSeek base URL
            client = OpenAI(api_key=api_key, base_url=api_url or DEFAULT_DEEPSEEK_API_URL)
            
            # Encode image to base64
            base64_image = encode_image_to_base64(image_path)
            
            # Create message with base64 image
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", 
                         "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]}
                ],
                temperature=0.2,
                max_tokens=2048
            )
            
            if response and response.choices and len(response.choices) > 0:
                raw_response = response.choices[0].message.content.strip()
            else:
                raise Exception("Invalid response from DeepSeek API")
        else:
            raise ValueError(f"Unsupported provider: {provider}")
            
        # Try to parse JSON from response
        try:
            # Extract JSON content from the response (it might be wrapped in markdown code blocks)
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', raw_response)
            if json_match:
                json_content = json_match.group(1)
            else:
                # If not in a code block, try to find JSON directly
                json_content = re.search(r'(\{[\s\S]*\})', raw_response).group(1)
            
            data = json.loads(json_content)
            return data
        except (json.JSONDecodeError, AttributeError):
            # If parsing fails, return raw response for manual processing
            console.print(f"[yellow]Failed to parse JSON from model response. Raw response:[/yellow]")
            console.print(raw_response)
            return {
                "date": "unknown",
                "sender": "unknown",
                "recipients": ["unknown"],
                "platform": "unknown",
                "content": raw_response,
                "context": "JSON parsing failed",
                "raw_response": raw_response
            }
    except Exception as e:
        console.print(f"[bold red]Error extracting data from {os.path.basename(image_path)}: {str(e)}[/bold red]")
        return {
            "date": "unknown",
            "sender": "unknown",
            "recipients": ["unknown"],
            "platform": "unknown",
            "content": f"ERROR: {str(e)}",
            "context": "Exception during processing",
            "error": str(e)
        }

async def analyze_communications(message_data: List[Dict[str, Any]], model: str, provider: str, api_key: str = None, api_url: str = None) -> str:
    """Analyze collected message data and create a chronological transcript"""
    try:
        # Create message with JSON data
        json_data = json.dumps(message_data, indent=2)
        content = f"{ANALYSIS_PROMPT}\n\nHere's the JSON data extracted from the screenshots:\n\n```json\n{json_data}\n```"
        
        console.print("[yellow]Analyzing all communications to create chronological transcript...[/yellow]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Analyzing communications...", total=None)
            
            if provider == 'ollama':
                response = ollama.chat(
                    model=model,
                    messages=[{'role': 'user', 'content': content}],
                    options={"temperature": 0.3, "num_predict": 4096}
                )
                
                if response and 'message' in response and 'content' in response['message']:
                    return response['message']['content'].strip()
                else:
                    raise Exception("Invalid response from Ollama")
            
            elif provider == 'deepseek':
                if not api_key:
                    raise ValueError("DeepSeek API key is required")
                    
                # Initialize OpenAI client with DeepSeek base URL
                client = OpenAI(api_key=api_key, base_url=api_url or DEFAULT_DEEPSEEK_API_URL)
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    temperature=0.3,
                    max_tokens=4096
                )
                
                if response and response.choices and len(response.choices) > 0:
                    return response.choices[0].message.content.strip()
                else:
                    raise Exception("Invalid response from DeepSeek API")
            else:
                raise ValueError(f"Unsupported provider: {provider}")
    except Exception as e:
        console.print(f"[bold red]Error analyzing communications: {str(e)}[/bold red]")
        return f"ERROR: {str(e)}"

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

def check_deepseek_api(api_key: str, api_url: str) -> bool:
    """Check if the DeepSeek API key is valid"""
    try:
        client = OpenAI(api_key=api_key, base_url=api_url or DEFAULT_DEEPSEEK_API_URL)
        
        # Simple test request to verify API key works
        response = client.chat.completions.create(
            model="deepseek-chat",  # Use a simple text model for checking
            messages=[
                {"role": "user", "content": "Hello, this is a test"}
            ],
            max_tokens=10  # Minimal tokens to reduce cost
        )
        
        if response and response.choices and len(response.choices) > 0:
            return True
        return False
    except Exception as e:
        console.print(f"[yellow]DeepSeek API check failed: {str(e)}[/yellow]")
        return False

def check_requirements():
    """Check if required packages are installed and selected provider is available"""
    global DEFAULT_MODEL, DEFAULT_PROVIDER
    
    try:
        # Choose provider
        console.print("[bold]Select model provider:[/bold]")
        console.print("1. Ollama (local models)")
        if DEEPSEEK_SUPPORT:
            console.print("2. DeepSeek API (cloud-based)")
            provider_choice = Prompt.ask("Choose provider", choices=["1", "2"], default="1")
            DEFAULT_PROVIDER = "ollama" if provider_choice == "1" else "deepseek"
        else:
            console.print("[dim]2. DeepSeek API (requires 'openai' package)[/dim]")
            provider_choice = "1"
            DEFAULT_PROVIDER = "ollama"
            
        # Check provider-specific requirements
        if DEFAULT_PROVIDER == 'ollama':
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
        
        elif DEFAULT_PROVIDER == 'deepseek':
            # Check for OpenAI package
            if not DEEPSEEK_SUPPORT:
                console.print("[bold red]Error: OpenAI package required for DeepSeek API.[/bold red]")
                console.print("[yellow]Please install with: pip install openai[/yellow]")
                sys.exit(1)
                
            # Get API key
            use_env_var = False
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            api_url = os.environ.get("DEEPSEEK_API_URL", DEFAULT_DEEPSEEK_API_URL)
            
            if api_key:
                if Confirm.ask(f"Found DEEPSEEK_API_KEY in environment. Use it?", default=True):
                    use_env_var = True
                    console.print("[green]✓ Using API key from environment variable[/green]")
            
            if not use_env_var:
                api_key = Prompt.ask("Enter your DeepSeek API key", password=True)
                custom_url = Confirm.ask("Use custom API URL? (default is https://api.deepseek.com)", default=False)
                if custom_url:
                    api_url = Prompt.ask("Enter DeepSeek API URL", default=DEFAULT_DEEPSEEK_API_URL)
            
            # Validate API key
            console.print("[yellow]Validating DeepSeek API credentials...[/yellow]")
            if check_deepseek_api(api_key, api_url):
                console.print("[green]✓ DeepSeek API connection successful[/green]")
                # Save for later use
                os.environ["DEEPSEEK_API_KEY"] = api_key
                os.environ["DEEPSEEK_API_URL"] = api_url
                DEFAULT_MODEL = DEFAULT_DEEPSEEK_MODEL
            else:
                console.print("[bold red]Error: Could not validate DeepSeek API credentials[/bold red]")
                if not Confirm.ask("Continue anyway?"):
                    sys.exit(1)
                    
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        sys.exit(1)

async def select_files_for_analysis(directory: str) -> List[str]:
    """Allow user to select which files to include in analysis"""
    all_files = [f for f in os.listdir(directory) if is_supported_file(os.path.join(directory, f))]
    
    if not all_files:
        console.print("[yellow]No supported image files found in this directory.[/yellow]")
        return []
    
    # Sort files by name for easier selection
    all_files.sort()
    
    # Create table of files
    table = Table(title=f"Screenshots in {directory}")
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Filename", style="green")
    
    for i, filename in enumerate(all_files, 1):
        table.add_row(str(i), filename)
    
    console.print(table)
    
    # Options for selection
    console.print("\n[bold]Selection options:[/bold]")
    console.print("1. Select all files")
    console.print("2. Select files by number (comma-separated)")
    console.print("3. Select files by pattern match")
    
    choice = Prompt.ask("Choose an option", choices=["1", "2", "3"], default="1")
    
    if choice == "1":
        return [os.path.join(directory, f) for f in all_files]
    
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
                
                return [os.path.join(directory, all_files[i-1]) for i in valid_indices]
            
            except ValueError:
                console.print("[red]Invalid input format. Please try again.[/red]")
    
    elif choice == "3":
        pattern = Prompt.ask("Enter filename pattern (e.g. 'board' or 'email')")
        matched_files = [f for f in all_files if pattern.lower() in f.lower()]
        
        if not matched_files:
            console.print(f"[yellow]No files matched the pattern '{pattern}'.[/yellow]")
            return await select_files_for_analysis(directory)
        
        console.print(f"[green]Selected {len(matched_files)} files matching '{pattern}':[/green]")
        for f in matched_files:
            console.print(f"  - {f}")
        
        if Confirm.ask("Use these files?", default=True):
            return [os.path.join(directory, f) for f in matched_files]
        else:
            return await select_files_for_analysis(directory)
    
    return []

async def process_screenshots(file_paths: List[str], prompt: str, model: str, provider: str) -> List[Dict[str, Any]]:
    """Process screenshots and extract message data"""
    if not file_paths:
        console.print("[yellow]No files selected for processing.[/yellow]")
        return []
    
    # Get API credentials if using DeepSeek
    api_key = None
    api_url = None
    if provider == 'deepseek':
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        api_url = os.environ.get("DEEPSEEK_API_URL", DEFAULT_DEEPSEEK_API_URL)
    
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
        main_task = progress.add_task(f"[cyan]Processing {len(file_paths)} screenshots...", total=len(file_paths), status="")
        
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            progress.update(main_task, description=f"[cyan]Processing {filename}...", status="")
            
            try:
                # Extract message data
                data = await extract_message_data(file_path, prompt, model, provider, api_key, api_url)
                
                # Add filename to data for reference
                data["filename"] = filename
                data["file_path"] = file_path
                
                results.append(data)
                progress.update(main_task, advance=1, status="Complete")
                
            except Exception as e:
                console.print(f"[bold red]Error processing {filename}: {str(e)}[/bold red]")
                results.append({
                    "date": "unknown",
                    "sender": "unknown",
                    "recipients": ["unknown"],
                    "platform": "unknown",
                    "content": f"ERROR: {str(e)}",
                    "context": "Exception during processing",
                    "filename": filename,
                    "file_path": file_path,
                    "error": str(e)
                })
                progress.update(main_task, advance=1, status="Error")
    
    return results

async def save_transcript(transcript: str, directory: str) -> str:
    """Save transcript to markdown file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(directory, f"board_transcript_{timestamp}.md")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(transcript)
    
    return output_path

async def save_extracted_data(data: List[Dict[str, Any]], directory: str) -> str:
    """Save extracted data to JSON file for future reference"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(directory, f"extracted_data_{timestamp}.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    return output_path

def display_data_summary(data: List[Dict[str, Any]]) -> Table:
    """Display summary of extracted data"""
    table = Table(title="Extracted Message Data")
    table.add_column("Date", style="cyan")
    table.add_column("Sender", style="green")
    table.add_column("Platform", style="magenta")
    table.add_column("Message Preview", style="yellow")
    
    # Sort data by date if possible
    sorted_data = sorted(
        data, 
        key=lambda x: x.get("date", "unknown"),
        # Unknown dates go to the end
        reverse=False
    )
    
    for item in sorted_data:
        date = item.get("date", "unknown")
        sender = item.get("sender", "unknown")
        platform = item.get("platform", "unknown")
        content = item.get("content", "")
        
        # Truncate content for preview
        preview = content[:50] + "..." if len(content) > 50 else content
        
        table.add_row(date, sender, platform, preview)
    
    return table

# Modify the main function to properly handle model provider selection
async def main():
    """Main application flow"""
    try:
        clear()
        console.print(banner())
        
        # First, check for DeepSeek support
        if not DEEPSEEK_SUPPORT:
            console.print("[yellow]Note: DeepSeek API support requires the 'openai' package.[/yellow]")
            console.print("[yellow]To enable DeepSeek support, install with: pip install openai[/yellow]")
        
        # Initial provider selection
        console.print("\n[bold]Select model provider:[/bold]")
        console.print("1. Ollama (local models)")
        if DEEPSEEK_SUPPORT:
            console.print("2. DeepSeek API (cloud-based)")
            provider_choice = Prompt.ask("Choose provider", choices=["1", "2"], default="1")
            provider = "ollama" if provider_choice == "1" else "deepseek"
        else:
            console.print("[dim]2. DeepSeek API (requires 'openai' package - unavailable)[/dim]")
            provider = "ollama"
        
        # Setup provider-specific configuration
        if provider == "ollama":
            # Check if Ollama is running
            try:
                ollama.list()
                console.print("[green]✓ Connected to Ollama successfully[/green]")
            except Exception as e:
                console.print(f"[bold red]Error: Could not connect to Ollama: {str(e)}[/bold red]")
                console.print("[yellow]Make sure Ollama is installed and running.[/yellow]")
                console.print("[blue]Installation instructions: https://ollama.com/download[/blue]")
                return
            
            # Check for the vision model
            model = DEFAULT_MODEL
            if check_ollama_model(model):
                console.print(f"[green]✓ {model} model is available[/green]")
            else:
                # Try alternate model names
                alternate_models = ['llava', 'bakllava', 'llama3-vision']
                
                found_alt = False
                for alt_model in alternate_models:
                    if check_ollama_model(alt_model):
                        console.print(f"[yellow]! {model} not found, but {alt_model} is available[/yellow]")
                        if Confirm.ask(f"Would you like to use {alt_model} instead?"):
                            model = alt_model
                            console.print(f"[green]✓ Using {alt_model} model instead[/green]")
                            found_alt = True
                            break
                
                if not found_alt:
                    console.print(f"[yellow]! {model} model not found. You may need to run:[/yellow]")
                    console.print(f"[blue]  ollama pull {model}[/blue]")
                    if not Confirm.ask("Continue anyway?"):
                        return
        
        elif provider == "deepseek":
            # Get API key
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            api_url = os.environ.get("DEEPSEEK_API_URL", DEFAULT_DEEPSEEK_API_URL)
            
            if api_key:
                if Confirm.ask("Found DEEPSEEK_API_KEY in environment. Use it?", default=True):
                    console.print("[green]✓ Using API key from environment variable[/green]")
                else:
                    api_key = Prompt.ask("Enter your DeepSeek API key", password=True)
            else:
                api_key = Prompt.ask("Enter your DeepSeek API key", password=True)
            
            # Ask about custom API URL
            if Confirm.ask("Use custom API URL? (default is https://api.deepseek.com)", default=False):
                api_url = Prompt.ask("Enter DeepSeek API URL", default=DEFAULT_DEEPSEEK_API_URL)
            
            # Save to environment for later use by process functions
            os.environ["DEEPSEEK_API_KEY"] = api_key
            os.environ["DEEPSEEK_API_URL"] = api_url
            
            # Validate API key
            console.print("[yellow]Validating DeepSeek API credentials...[/yellow]")
            if check_deepseek_api(api_key, api_url):
                console.print("[green]✓ DeepSeek API connection successful[/green]")
                model = DEFAULT_DEEPSEEK_MODEL
            else:
                console.print("[bold red]Error: Could not validate DeepSeek API credentials[/bold red]")
                if not Confirm.ask("Continue anyway?"):
                    return
                model = DEFAULT_DEEPSEEK_MODEL
        
        # Press Enter to continue
        console.print("\nPress Enter to continue to the main menu...")
        input()
        
        # Set up initial state
        current_directory = os.getcwd()
        prompt = DEFAULT_PROMPT
        
        # Main application loop
        while True:
            clear()
            console.print(banner())
            
            # Always display the menu options
            console.print("\n[bold cyan]Menu Options:[/bold cyan]")
            console.print("  1. Select directory containing screenshots")
            console.print("  2. Select screenshots for analysis")
            console.print("  3. Customize extraction prompt")
            console.print("  4. Process screenshots and extract data")
            console.print("  5. Generate transcript")
            console.print("  6. Change model provider") # New option
            console.print("  7. Exit")
            
            # Current status display
            console.print(f"\n[blue]Current directory:[/blue] {current_directory}")
            console.print(f"[blue]Current model:[/blue] {model}")
            console.print(f"[blue]Current provider:[/blue] {provider}")
            
            # Get user choice
            choice = Prompt.ask("\nChoose an option", choices=["1", "2", "3", "4", "5", "6", "7"], default="1")
            
            if choice == "1":
                # Step 1: Select directory with screenshots
                console.print("\n[bold]Step 1: Select directory containing screenshots[/bold]")
                try:
                    # Try the async version in a new event loop
                    current_directory = await interactive_directory_browser(current_directory)
                except Exception as e:
                    console.print(f"[yellow]Error with directory browser: {str(e)}[/yellow]")
            
            elif choice == "2":
                # Step 2: Select files for analysis
                console.print("\n[bold]Step 2: Select screenshots for analysis[/bold]")
                file_paths = await select_files_for_analysis(current_directory)
                
                if not file_paths:
                    console.print("[yellow]No files selected.[/yellow]")
                    continue
                
                console.print(f"[green]Selected {len(file_paths)} files for analysis.[/green]")
            
            elif choice == "3":
                # Step 3: Customize prompt if needed
                console.print("\n[bold]Step 3: Customize extraction prompt[/bold]")
                console.print("[blue]Current prompt:[/blue]")
                console.print(Panel(prompt, title="Extraction Prompt", border_style="blue"))
                
                # Show simplified explanation of what the prompt should accomplish
                console.print("[yellow]The prompt should instruct the model to extract:[/yellow]")
                console.print("- Date and time of message")
                console.print("- Sender and recipients")
                console.print("- Message content")
                console.print("- Platform (email, text, etc.)")
                console.print("- Format as JSON")
                
                new_prompt = Prompt.ask("Enter new prompt (or press Enter to keep current)", default=prompt)
                prompt = new_prompt
            
            elif choice == "4":
                # Step 4: Process screenshots and extract data
                console.print("\n[bold]Step 4: Process screenshots and extract data[/bold]")
                if not 'file_paths' in locals() or not file_paths:
                    console.print("[yellow]No files selected. Please select files first (option 2).[/yellow]")
                    continue
                    
                data = await process_screenshots(file_paths, prompt, model, provider)
                
                if not data:
                    console.print("[yellow]No data extracted.[/yellow]")
                    continue
                
                # Display summary of extracted data
                console.print(display_data_summary(data))
                
                # Save extracted data
                data_file = await save_extracted_data(data, current_directory)
                console.print(f"[green]Extracted data saved to: {data_file}[/green]")
            
            elif choice == "5":
                # Step 5: Generate transcript from extracted data
                console.print("\n[bold]Step 5: Generate chronological transcript[/bold]")
                
                if not 'data' in locals() or not data:
                    console.print("[yellow]No data extracted. Please process files first (option 4).[/yellow]")
                    continue
                
                # Get API credentials if using DeepSeek
                api_key = None
                api_url = None
                if provider == 'deepseek':
                    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
                    api_url = os.environ.get("DEEPSEEK_API_URL", DEFAULT_DEEPSEEK_API_URL)
                
                transcript = await analyze_communications(data, model, provider, api_key, api_url)
                
                if transcript.startswith("ERROR:"):
                    console.print(f"[bold red]{transcript}[/bold red]")
                    continue
                
                # Save transcript to file
                transcript_file = await save_transcript(transcript, current_directory)
                console.print(f"[green]Transcript saved to: {transcript_file}[/green]")
                
                # Display transcript preview
                console.print(Panel(
                    transcript[:1000] + ("..." if len(transcript) > 1000 else ""),
                    title="Transcript Preview (first 1000 characters)",
                    border_style="green"
                ))
                
                # Ask if user wants to open the transcript
                if Confirm.ask("\nOpen the transcript file?", default=True):
                    try:
                        if sys.platform == 'darwin':  # macOS
                            os.system(f'open "{transcript_file}"')
                        elif sys.platform == 'win32':  # Windows
                            os.system(f'start "" "{transcript_file}"')
                        else:  # Linux
                            os.system(f'xdg-open "{transcript_file}"')
                    except Exception as e:
                        console.print(f"[yellow]Could not open file: {str(e)}[/yellow]")
            
            elif choice == "6":
                # Change model provider
                console.print("\n[bold]Change model provider:[/bold]")
                console.print("1. Ollama (local models)")
                if DEEPSEEK_SUPPORT:
                    console.print("2. DeepSeek API (cloud-based)")
                    provider_choice = Prompt.ask("Choose provider", choices=["1", "2"], default="1")
                    new_provider = "ollama" if provider_choice == "1" else "deepseek"
                else:
                    console.print("[dim]2. DeepSeek API (requires 'openai' package - unavailable)[/dim]")
                    new_provider = "ollama"
                
                if new_provider != provider:
                    provider = new_provider
                    # Setup provider-specific configuration
                    if provider == "ollama":
                        # Check Ollama connection and model
                        try:
                            ollama.list()
                            console.print("[green]✓ Connected to Ollama successfully[/green]")
                            model = DEFAULT_MODEL
                            if not check_ollama_model(model):
                                console.print(f"[yellow]! {model} model not found, please check available models.[/yellow]")
                        except Exception as e:
                            console.print(f"[bold red]Error: Could not connect to Ollama: {str(e)}[/bold red]")
                    
                    elif provider == "deepseek":
                        # Get or update DeepSeek API key
                        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
                        api_url = os.environ.get("DEEPSEEK_API_URL", DEFAULT_DEEPSEEK_API_URL)
                        
                        if not api_key or not Confirm.ask("Use existing API key?", default=True):
                            api_key = Prompt.ask("Enter your DeepSeek API key", password=True)
                            os.environ["DEEPSEEK_API_KEY"] = api_key
                        
                        # Ask about custom API URL
                        if Confirm.ask("Use custom API URL?", default=False):
                            api_url = Prompt.ask("Enter DeepSeek API URL", default=DEFAULT_DEEPSEEK_API_URL)
                            os.environ["DEEPSEEK_API_URL"] = api_url
                        
                        # Set model to DeepSeek model
                        model = DEFAULT_DEEPSEEK_MODEL
                
                console.print(f"[green]Provider changed to {provider}.[/green]")
            
            elif choice == "7":
                console.print("[cyan]Exiting application.[/cyan]")
                break
                
            # Add a small pause between menu displays
            if choice != "7":
                console.print("\nPress Enter to continue...")
                input()
                
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user. Exiting...[/yellow]")
    except Exception as e:
        console.print(f"[bold red]Unexpected error: {str(e)}[/bold red]")
        import traceback
        console.print(traceback.format_exc())

if __name__ == "__main__":
    # Ensure proper event loop handling - fixes common asyncio issues
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Run the async main function directly
    asyncio.run(main())