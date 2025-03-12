#!/usr/bin/env python3

import re
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import pytesseract
from PIL import Image
from rich.console import Console
from rich.panel import Panel
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
"""

# Initialize Rich console
console = Console()

def load_config() -> Dict[str, Any]:
    """Load configuration from file"""
    default_config = {
        "last_directory": os.getcwd(),
        "output_directory": DEFAULT_OUTPUT_DIR,
        "last_language": "eng",
        "pre_prompt": "",
        "post_prompt": ""
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

def select_directory() -> str:
    """Simple directory selection via prompt"""
    console.print("[yellow]Enter directory path:[/yellow]")
    directory = input("> ")
    directory = os.path.expanduser(directory)
    
    if not os.path.isdir(directory):
        console.print("[red]Invalid directory.[/red]")
        return os.getcwd()
    
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

async def extract_text_from_image(image_path: str, lang: str = "eng") -> str:
    """Extract text from image using OCR"""
    try:
        if image_path.lower().endswith('.pdf'):
            from pdf2image import convert_from_path
            pages = convert_from_path(image_path, 300)
            text_results = [pytesseract.image_to_string(page, lang=lang) for page in pages]
            combined_text = "\n\n".join(text_results)
        else:
            preprocessed_image = preprocess_image(image_path)
            
            # Use pytesseract
            configs = ['--psm 6 --oem 3', '--psm 4 --oem 3', '--psm 3 --oem 3']
            tesseract_texts = [pytesseract.image_to_string(preprocessed_image, lang=lang, config=cfg) for cfg in configs]
            tesseract_text = max(tesseract_texts, key=len)
            
            # Combine the results
            combined_text = tesseract_text

        # Post-process and clean with Ollama
        cleaned_text = await clean_text_with_ollama(combined_text)

        return post_process_ocr_text(cleaned_text)
    except Exception as e:
        console.print(f"[bold red]Error processing {image_path}: {str(e)}[/bold red]")
        return f"ERROR: {str(e)}"

async def clean_text_with_ollama(text: str) -> str:
    """Clean the OCR text using Ollama."""
    try:
        import ollama
        prompt = f"Correct and clean the following OCR text:\n\n{text}\n\nCleaned Text:"

        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1}
        )

        cleaned_text = response.message.content if hasattr(response, 'message') and hasattr(response.message, 'content') else response["message"]["content"]
        return cleaned_text
    except ImportError:
        console.print("[yellow]Ollama not installed. Install with: pip install ollama[/yellow]")
        return text
    except Exception as e:
        console.print(f"[yellow]Error cleaning text with Ollama: {str(e)}[/yellow]")
        return text

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
                ocr_text = await extract_text_from_image(file_path, lang)
                metadata = get_image_metadata(file_path)
                
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

async def analyze_with_ollama(results_file: str, output_dir: str):
    """Analyze OCR results with Ollama using llama3"""
    try:
        import ollama

        console.print("[cyan]Connecting to Ollama server...[/cyan]")

        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            results = data.get("results", [])

        file_paths = [result.get("file_path") for result in results if "file_path" in result]
        image_map = copy_images_to_output(file_paths, output_dir)

        markdown_path = await create_markdown_summary(results, results_file, image_map)
        console.print(f"[green]✓ Created initial markdown summary: {markdown_path}[/green]")

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

        post_prompt_file = "postprompt.txt"
        post_prompt = ""
        if os.path.exists(post_prompt_file):
            try:
                with open(post_prompt_file, 'r', encoding='utf-8') as f:
                    post_prompt = f.read()
                console.print(f"[green]✓ Loaded custom post-prompt from {post_prompt_file}[/green]")
            except Exception as e:
                console.print(f"[yellow]Error reading {post_prompt_file}: {str(e)}[/yellow]")
                console.print("[yellow]Continuing with default post-prompt...[/yellow]")

        prompt = ANALYSIS_PROMPT + "\n\nHere's the OCR data extracted from screenshots:\n"
        prompt += json.dumps(simplified_data, indent=2)
        prompt += "\n\n" + post_prompt

        model_name = "llama3.2"

        console.print(f"[cyan]Checking Ollama model: {model_name}[/cyan]")
        try:
            models = ollama.list()
            model_names = [m.get("name") for m in models.models] if hasattr(models, 'models') else [m.get("name") for m in models["models"]]

            if model_name not in model_names:
                console.print(f"[yellow]Model {model_name} not found. Pulling from repository...[/yellow]")
                ollama.pull(model_name)
        except Exception as e:
            console.print(f"[yellow]Error checking models, will try to use {model_name} directly: {str(e)}[/yellow]")

        console.print(f"[cyan]Analyzing text with {model_name}...[/cyan]")
        start_time = time.time()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(complete_style="cyan", finished_style="green"),
            TextColumn("[cyan]{task.fields[status]}"),
            TimeElapsedColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console
        ) as progress:
            analysis_task = progress.add_task(f"[cyan]Analyzing text with {model_name}...", total=100, status="")

            response = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.2,
                    "num_predict": 4096
                }
            )

            analysis = response.message.content if hasattr(response, 'message') and hasattr(response.message, 'content') else response["message"]["content"]
            progress.update(analysis_task, advance=100, status="Complete")

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

        elapsed_time = time.time() - start_time
        console.print(f"[green]✓ Analysis completed in {elapsed_time:.2f} seconds[/green]")
        console.print(f"[green]✓ Results saved to: {output_path}[/green]")

        os.system(f'open "{output_path}"')

    except ImportError:
        console.print("[red]Ollama package not installed. Install with: pip install ollama[/red]")
    except Exception as e:
        console.print(f"[bold red]Error during analysis: {str(e)}[/bold red]")

def get_available_models() -> List[str]:
    """Get available models from Ollama"""
    try:
        import ollama
        models = ollama.list()
        return [m.get("name") for m in models.models] if hasattr(models, 'models') else [m.get("name") for m in models["models"]]
    except Exception as e:
        console.print(f"[yellow]Error fetching models: {str(e)}[/yellow]")
        return []

def config_menu(config: Dict[str, Any]) -> Dict[str, Any]:
    """Display and handle the configuration menu"""
    while True:
        console.print("\n[bold cyan]Configuration Menu:[/bold cyan]")
        console.print("  1. Set pre-prompt")
        console.print("  2. Set post-prompt")
        console.print("  3. View available models")
        console.print("  4. Back to main menu")
        
        choice = Prompt.ask("\nChoose an option", choices=["1", "2", "3", "4"], default="4")
        
        if choice == "1":
            config["pre_prompt"] = Prompt.ask("Enter pre-prompt", default=config.get("pre_prompt", ""))
            save_config(config)
            console.print("[green]Pre-prompt updated.[/green]")
        
        elif choice == "2":
            config["post_prompt"] = Prompt.ask("Enter post-prompt", default=config.get("post_prompt", ""))
            save_config(config)
            console.print("[green]Post-prompt updated.[/green]")
        
        elif choice == "3":
            models = get_available_models()
            if models:
                console.print("[blue]Available models:[/blue]")
                for model in models:
                    console.print(f"  - {model}")
            else:
                console.print("[red]No models available or Ollama not reachable.[/red]")
        
        elif choice == "4":
            break
    
    return config

async def main():
    """Main application flow"""
    try:
        console.print(banner())
        check_tesseract_installation()
        
        config = load_config()
        current_directory = config.get("last_directory", os.getcwd())
        output_dir = config.get("output_directory", DEFAULT_OUTPUT_DIR)
        lang = config.get("last_language", "eng")
        
        while True:
            console.print("\n[bold cyan]Menu Options:[/bold cyan]")
            console.print("  1. Select directory with images")
            console.print("  2. Process all images in directory")
            console.print("  3. Configuration")
            console.print("  4. Exit")
            
            console.print(f"\n[blue]Current directory:[/blue] {current_directory}")
            console.print(f"[blue]Output directory:[/blue] {output_dir}")
            console.print(f"[blue]OCR language:[/blue] {lang}")
            
            choice = Prompt.ask("\nChoose an option", choices=["1", "2", "3", "4"], default="1")
            
            if choice == "1":
                prev_dir = current_directory
                current_directory = select_directory()
                if prev_dir != current_directory:
                    config["last_directory"] = current_directory
                    save_config(config)
            
            elif choice == "2":
                console.print(f"\n[bold]Processing images in: {current_directory}[/bold]")
                
                if not os.path.isdir(current_directory):
                    console.print("[red]Directory doesn't exist![/red]")
                    continue
                    
                file_paths = get_all_images_in_directory(current_directory)
                if not file_paths:
                    console.print("[yellow]No supported image files found in this directory.[/yellow]")
                    continue
                
                console.print(f"[green]Found {len(file_paths)} image files.[/green]")
                
                results = await process_images(file_paths, lang)
                
                if not results:
                    console.print("[yellow]No results extracted.[/yellow]")
                    continue
                
                results_file = await save_results(results, output_dir)
                console.print(f"[green]OCR results saved to: {results_file}[/green]")
                
                console.print(display_result_summary(results))
                
                await analyze_with_ollama(results_file, output_dir)
            
            elif choice == "3":
                config = config_menu(config)
            
            elif choice == "4":
                console.print("[cyan]Exiting application.[/cyan]")
                
                config["last_directory"] = current_directory
                config["output_directory"] = output_dir
                config["last_language"] = lang
                save_config(config)
                
                return
                
            if choice != "4":
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


