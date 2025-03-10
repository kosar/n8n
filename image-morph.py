#!/usr/bin/env python3

import os
import sys
import asyncio
import time
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
import ollama
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn
from rich.table import Table
from rich.prompt import Prompt
from rich.layout import Layout
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import PathCompleter

# Default Settings - Now configurable via menu
DEFAULT_MODEL = 'llama3.2-vision'
DEFAULT_SCREENSHOT_ANALYSIS_PROMPT = """
Analyze this image as a screenshot of a conversation between board members about a dog bite incident at the "11 West" building.
Identify each message, the sender, the recipient (if explicitly mentioned or inferable), the date, and the time of the message.

Try to infer the date even if it is abbreviated or partially visible, or mentioned in a relative context (e.g., "yesterday"). If full date is not available, note down any partial date information.
Assume the conversation is about a dog bite incident involving the live-in superintendent's dog biting a worker of a building resident at "11 West".

Output each message as a structured item with the following fields if you can identify them:
- Sender: (Name of the sender, if identifiable)
- Recipient: (Name of the recipient, if explicitly mentioned or inferable)
- Date: (Full date of the message, as YYYY-MM-DD if possible, or partial date if full date is not available)
- Time: (Time of the message, in HH:MM format, 24-hour clock if possible, or 12-hour with AM/PM if that's what's visible)
- Message: (The text content of the message)

If the date and time are clearly associated with a group of messages (e.g., a date header above a series of messages), apply that date and time context to those messages appropriately.

Try to return each message as a distinct structured entry, even if they are visually grouped together in the screenshot. Be as complete and accurate as possible based on the image content.

If some information is not available, indicate it as "Unknown" or "Not identifiable".

Format your output in a way that is easy to parse, preferably line by line with each line representing a communication like:
'Sender: [Sender Name], Recipient: [Recipient Name], Date: [YYYY-MM-DD or Partial Date], Time: [HH:MM or Partial Time], Message: [Message Text]'
or a similar easily parseable line-based format.
"""
DEFAULT_BUILDING_NAME = "11 West"
DEFAULT_INCIDENT_TYPE = "dog bite"
DEFAULT_INCIDENT_DETAILS = "live-in superintendent's dog bit a worker who worked for a building resident"
DEFAULT_BOARD_MEMBER_GROUP = "board of directors"


# Global Settings Dictionary - User configurable settings are stored here
settings = {
    "model": DEFAULT_MODEL,
    "screenshot_analysis_prompt": DEFAULT_SCREENSHOT_ANALYSIS_PROMPT,
    "building_name": DEFAULT_BUILDING_NAME,
    "incident_type": DEFAULT_INCIDENT_TYPE,
    "incident_details": DEFAULT_INCIDENT_DETAILS,
    "board_member_group": DEFAULT_BOARD_MEMBER_GROUP,
}


# Constants
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}


# Initialize Rich console
console = Console()
layout = Layout() # Initialize layout

# Define layout structure globally, before any function that uses it
layout.split(
    Layout(name="header", size=10),
    Layout(name="body"),
)
layout["header"].split_row(
    Layout(name="status", ratio=3),
    Layout(name="menu", ratio=2),
)


def banner():
    """Display application banner with settings-driven building and incident info"""
    return Panel.fit(
        f"[bold blue]Screenshot Transcript Generator for {settings['building_name']} {settings['incident_type']} Incident[/bold blue]",
        border_style="green"
    )

def update_layout(current_directory: str, selected_files: Set[str]):
    """Update the fixed header with current status and menu"""
    # Status panel
    status_content = f"[blue]Current directory:[/blue] {current_directory}\n"
    status_content += f"[blue]Selected files:[/blue] {len(selected_files)}\n"
    status_content += f"[blue]Current model:[/blue] {settings['model']}"

    layout["status"].update(Panel(status_content, title="Status", border_style="blue"))

    # Menu panel
    menu_content = "[bold cyan]Menu Options:[/bold cyan]\n"
    menu_content += "  1. Change directory\n"
    menu_content += "  2. View and select image files\n"
    menu_content += "  3. Process selected images\n"
    menu_content += "  4. Edit Analysis Prompt\n" # New Menu Option
    menu_content += "  5. Settings\n" # Added Settings Menu
    menu_content += "  6. Exit"

    layout["menu"].update(Panel(menu_content, title="Menu", border_style="cyan"))


def is_image_file(file_path: str) -> bool:
    """Check if a file is a supported image type"""
    return os.path.isfile(file_path) and os.path.splitext(file_path)[1].lower() in SUPPORTED_EXTENSIONS

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
        update_layout(current_dir, set())  # Update layout at the start of each loop
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


async def analyze_screenshot(image_path: str) -> List[Dict[str, str]]:
    """Analyzes a single screenshot image using Ollama vision model, now using settings prompt."""
    console.print(f"[cyan]Analyzing screenshot:[/cyan] {os.path.basename(image_path)}")
    try:
        response_stream = await ollama.chat(
            model=settings["model"], # Use model from settings
            messages=[
                {
                    'role': 'user',
                    'content': settings["screenshot_analysis_prompt"], # Use prompt from settings
                    'images': [image_path],
                }
            ],
            stream=True
        )
        collected_response = ""
        async for part in response_stream:
            collected_response += part['message']['content']
        console.print(Panel(collected_response, title=f"Analysis of {os.path.basename(image_path)}", border_style="blue"))
        return parse_ollama_response(collected_response)

    except Exception as e:
        console.print(f"[bold red]Error analyzing {os.path.basename(image_path)}: {e}[/bold red]")
        return []

def parse_ollama_response(response_text: str) -> List[Dict[str, str]]:
    """Parses the raw text response from Ollama."""
    communications = []
    for line in response_text.strip().split('\n'):
        parts = line.split(', ')
        communication = {
            "Sender": "Unknown",
            "Recipient": "Unknown",
            "Date": "Unknown",
            "Time": "Unknown",
            "Message": "Unknown"
        }
        for part in parts:
            if ":" in part:
                key, value = part.split(":", 1)
                communication[key.strip()] = value.strip()
        if communication["Message"] != "Unknown":
            communications.append(communication)
    return communications

def sort_communications_chronologically(communications: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Sorts communications chronologically."""
    def get_sort_key(comm):
        date_str = comm.get("Date", "Unknown")
        time_str = comm.get("Time", "Unknown")
        try:
            if date_str != "Unknown" and time_str != "Unknown":
                datetime_obj = None
                for date_format in ["%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%Y/%m/%d", "%d-%m-%Y", "%d/%m/%y", "%d/%m/%Y", "%b %d, %Y"]:
                    for time_format in ["%H:%M", "%I:%M%p", "%I:%M %p", "%H.%M"]:
                        try:
                            datetime_obj = datetime.strptime(f"{date_str} {time_str}", f"{date_format} {time_format}")
                            return datetime_obj
                        except ValueError:
                            continue
                if datetime_obj:
                    return datetime_obj

        except ValueError:
            pass
        return datetime.max

    return sorted(communications, key=get_sort_key)


def generate_markdown_transcript(communications: List[Dict[str, str]], folder_name: str) -> str:
    """Generates markdown transcript, using settings-driven building and incident info."""
    markdown_content = f"# Chronological Communication Transcript for '{folder_name}' Screenshots\n\n"
    markdown_content += f"## Incident: {settings['incident_type']} at {settings['building_name']} involving {settings['incident_details']}\n\n" # Settings used here
    markdown_content += f"### {settings['board_member_group']} Communications\n\n" # Settings used here

    if not communications:
        markdown_content += "No communications extracted from screenshots.\n"
        return markdown_content

    for comm in communications:
        markdown_content += "---\n"
        markdown_content += f"**Sender:** {comm.get('Sender', 'Unknown')}\n\n"
        if comm.get('Recipient', 'Unknown') != 'Unknown':
            markdown_content += f"**Recipient:** {comm.get('Recipient', 'Unknown')}\n\n"
        markdown_content += f"**Date:** {comm.get('Date', 'Unknown')}\n\n"
        markdown_content += f"**Time:** {comm.get('Time', 'Unknown')}\n\n"
        markdown_content += f"**Message:** {comm.get('Message', 'Unknown')}\n\n"

    markdown_content += "---\n\n"
    markdown_content += "*End of Transcript*\n"
    return markdown_content

def save_transcript_to_file(markdown_transcript: str, folder_path: str) -> str:
    """Saves transcript to file."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_name = f"transcript_{os.path.basename(folder_path)}_{timestamp}.md"
    transcript_path = os.path.join(folder_path, file_name)
    try:
        with open(transcript_path, "w") as f:
            f.write(markdown_transcript)
        return transcript_path
    except Exception as e:
        console.print(f"[bold red]Error saving transcript to file: {e}[/bold red]")
        return None


async def process_screenshots_in_folder(folder_path: str):
    """Process screenshots in folder, analyze, and generate transcript."""
    console.print(f"[blue]Selected folder:[/blue] {folder_path}")
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if is_image_file(os.path.join(folder_path, f))]

    if not image_files:
        console.print("[yellow]No supported image files found in this folder.[/yellow]")
        return

    console.print(f"[blue]Found {len(image_files)} image files to process.[/blue]")
    all_communications = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        BarColumn(complete_style="cyan", finished_style="green"),
        TextColumn("[cyan]{task.fields[status]}"),
        transient=False,
    ) as progress:
        task_id = progress.add_task("[cyan]Analyzing screenshots...", total=len(image_files), status="")

        for image_path in image_files:
            progress.update(task_id, description=f"[cyan]Analyzing {os.path.basename(image_path)}...", status="")
            communications = await analyze_screenshot(image_path) # No prompt argument here, using settings
            if communications:
                all_communications.extend(communications)
            progress.update(task_id, advance=1, status="Complete")

    console.print("[green]Screenshot analysis complete.[/green]")

    if all_communications:
        console.print("[blue]Sorting communications chronologically...[/blue]")
        chronologically_sorted_communications = sort_communications_chronologically(all_communications)

        console.print("[blue]Generating markdown transcript...[/blue]")
        markdown_transcript = generate_markdown_transcript(chronologically_sorted_communications, os.path.basename(folder_path))

        transcript_file_path = save_transcript_to_file(markdown_transcript, folder_path)
        if transcript_file_path:
            console.print(Panel(f"[green]Transcript generated and saved to:[/green] [bold]{transcript_file_path}[/bold]", title="Transcript Generation", border_style="green"))
        else:
            console.print("[bold red]Failed to save transcript to file, but transcript content is:[/bold red]\n")
            console.print(markdown_transcript)
    else:
        console.print("[yellow]No communication data extracted from any screenshots.[/yellow]")

async def edit_analysis_prompt():
    """Allows user to edit the screenshot analysis prompt."""
    console.clear()
    console.print(banner())
    update_layout(os.getcwd(), set()) # Update layout after banner
    current_prompt = settings["screenshot_analysis_prompt"]
    console.print("[blue]Current Analysis Prompt:[/blue]")
    console.print(Panel(current_prompt, border_style="blue"))
    console.print("[yellow]Enter new analysis prompt below (or leave empty to keep current):[/yellow]")

    session = PromptSession()
    new_prompt = await session.prompt_async("New prompt> ", multiline=True) # Use multiline prompt

    if new_prompt.strip():
        settings["screenshot_analysis_prompt"] = new_prompt.strip()
        console.print("[green]Analysis prompt updated successfully![/green]")
    else:
        console.print("[yellow]Prompt not changed.[/yellow]")
    time.sleep(1) # Small pause to read message before menu redraw

async def display_settings():
    """Displays current settings to the user."""
    console.clear()
    console.print(banner())
    update_layout(os.getcwd(), set()) # Update layout after banner
    settings_table = Table(title="Current Settings", show_header=False)
    for key, value in settings.items():
        settings_table.add_row(f"[cyan]{key}[/cyan]", str(value)) # Display settings in a table

    console.print(settings_table)
    console.print("\n[yellow]Press Enter to return to the main menu[/yellow]")
    input() # Wait for Enter press

async def settings_menu():
    """Interactive menu for displaying and changing application settings."""
    while True:
        console.clear()
        console.print(banner())
        update_layout(os.getcwd(), set()) # Update layout after banner and before menu
        console.print("[bold magenta]Application Settings Menu[/bold magenta]")
        await display_settings() # Display current settings

        console.print("\n[bold cyan]Settings Options:[/bold cyan]")
        console.print("  1. Change Model")
        console.print("  2. Change Building Name")
        console.print("  3. Change Incident Type")
        console.print("  4. Change Incident Details")
        console.print("  5. Change Board Member Group Name")
        console.print("  6. Back to Main Menu")

        choice = Prompt.ask("Choose an option", choices=['1', '2', '3', '4', '5', '6'], default='6')

        if choice == '1':
            new_model = Prompt.ask("Enter new model name", default=settings['model'])
            settings['model'] = new_model.strip()
        elif choice == '2':
            new_building_name = Prompt.ask("Enter new building name", default=settings['building_name'])
            settings['building_name'] = new_building_name.strip()
        elif choice == '3':
            new_incident_type = Prompt.ask("Enter new incident type", default=settings['incident_type'])
            settings['incident_type'] = new_incident_type.strip()
        elif choice == '4':
            new_incident_details = Prompt.ask("Enter new incident details", default=settings['incident_details'], multiline=True) # Multiline for details
            settings['incident_details'] = new_incident_details.strip()
        elif choice == '5':
            new_board_member_group = Prompt.ask("Enter new board member group name", default=settings['board_member_group'])
            settings['board_member_group'] = new_board_member_group.strip()
        elif choice == '6':
            break # Exit settings menu

        console.print("[green]Setting updated.[/green]")
        time.sleep(1) # Pause to read confirmation


async def main_menu():
    """Main application menu."""
    current_directory = os.getcwd()
    selected_files: Set[str] = set()

    # Initialize layout with named regions HERE, moved to global scope for correct initialization
    # layout.split(
    #     Layout(name="header", size=10),
    #     Layout(name="body"),
    # )
    # layout["header"].split_row(
    #     Layout(name="status", ratio=3),
    #     Layout(name="menu", ratio=2),
    # )


    while True:
        console.clear()
        console.print(banner())
        update_layout(current_directory, selected_files) # Update header layout

        choice = Prompt.ask("Enter menu option", choices=['1', '2', '3', '4', '5', '6'], default='2')

        if choice == '1':
            current_directory = await interactive_directory_browser(current_directory)
        elif choice == '2':
            selected_files_list = [os.path.join(current_directory, f) for f in os.listdir(current_directory) if is_image_file(os.path.join(current_directory, f))]
            if not selected_files_list:
                console.print("[yellow]No image files in the current directory to select from.[/yellow]")
                time.sleep(1)
                continue # Go back to main menu directly
            selected_files = set(selected_files_list) # Select all images in the dir for simplicity now.
            console.print(f"[green]All image files in '{current_directory}' selected for processing.[/green]")

        elif choice == '3':
            if selected_files:
                await process_screenshots_in_folder(current_directory) # Process all selected (which is all in dir now)
                selected_files = set() # Clear selection after processing in this simplified flow
            else:
                console.print("[yellow]No files selected. Please select files first (option 2).[/yellow]")
                time.sleep(1)
        elif choice == '4':
            await edit_analysis_prompt() # Edit prompt option
        elif choice == '5':
            await settings_menu() # Open settings menu
        elif choice == '6':
            console.print("[cyan]Exiting application.[/cyan]")
            break
        else:
            console.print("[red]Invalid choice. Please select a valid menu option.[/red]")
            time.sleep(1) # Small delay for error msg

async def main():
    await main_menu() # Start the main menu loop

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold red]Aborted by user.[/bold red]")
    except Exception as e:
        console.print(f"\n[bold red]An unexpected error occurred:[/bold red] {e}")