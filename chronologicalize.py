#!/usr/bin/env python3

import os
import sys
import asyncio
from typing import List, Set
import ollama
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.layout import Layout
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter

# Constants (You can adjust these if needed)
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
DEFAULT_MODEL = 'llama3.2-vision'
DEFAULT_PROMPT = "Describe what you see in this image in detail." # We'll change this later

# Initialize Rich console
console = Console()
layout = Layout() # Initialize layout (if you want to use it later)

def banner():
    """Display a simple application banner"""
    return Panel.fit(
        "[bold blue]Screenshot Transcript Generator[/bold blue]",
        border_style="green"
    )

def is_image_file(file_path: str) -> bool:
    """Check if a file is a supported image type"""
    return os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in SUPPORTED_EXTENSIONS

async def interactive_directory_browser(start_dir: str = os.getcwd()) -> str:
    """Interactive directory browser with autocomplete (from vision_explorer.py)"""
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

            if user_input == "..":
                current_dir = os.path.dirname(current_dir)
                continue

            if os.path.isabs(user_input):
                path = user_input
            else:
                path = os.path.join(current_dir, user_input)

            path = os.path.normpath(os.path.expanduser(path))

            if os.path.isdir(path):
                current_dir = path
            else:
                console.print("[bold red]Not a valid directory. Please try again.[/bold red]")

        except KeyboardInterrupt:
            raise
        except Exception as e:
            console.print(f"[bold red]Error: {str(e)}[/bold red]")

async def process_screenshots_in_folder(folder_path: str):
    """Process all screenshots in the selected folder."""
    console.print(f"[blue]Selected folder:[/blue] {folder_path}")
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if is_image_file(os.path.join(folder_path, f))]

    if not image_files:
        console.print("[yellow]No supported image files found in this folder.[/yellow]")
        return

    console.print(f"[blue]Found {len(image_files)} image files to process.[/blue]")
    # In the next steps, we will add the code to process each image file
    # and generate the transcript here.
    for image_file in image_files:
        console.print(f"[cyan]Preparing to process:[/cyan] {os.path.basename(image_file)}")
        # ... processing logic will go here ...


async def main():
    console.clear()
    console.print(banner())

    screenshot_folder = await interactive_directory_browser()
    if screenshot_folder:
        await process_screenshots_in_folder(screenshot_folder)
    else:
        console.print("[yellow]No folder selected. Exiting.[/yellow]")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]Aborted by user.[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred:[/bold red] {e}")