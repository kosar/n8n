#!/usr/bin/env python3

import os
import sys
import argparse
import asyncio
from date_extractor import (
    extract_date_from_image, validate_and_format_date, rename_file_with_date,
    is_image_file, is_pdf_file, convert_pdf_to_images, process_directory,
    DEFAULT_MODEL, DEFAULT_PROMPT
)

async def main():
    """Command-line interface for date extraction"""
    parser = argparse.ArgumentParser(
        description="Extract dates from images and PDFs using vision AI and rename files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "directory",
        help="Directory containing images/PDFs to process",
        type=str,
        nargs="?",
        default=os.getcwd()
    )
    
    parser.add_argument(
        "-r", "--recursive",
        help="Process subdirectories recursively",
        action="store_true"
    )
    
    parser.add_argument(
        "-m", "--model",
        help="Ollama vision model to use",
        type=str,
        default=DEFAULT_MODEL
    )
    
    parser.add_argument(
        "-p", "--prompt",
        help="Prompt for date extraction",
        type=str,
        default=DEFAULT_PROMPT
    )
    
    parser.add_argument(
        "-d", "--dry-run",
        help="Don't rename files, just show what would happen",
        action="store_true"
    )
    
    parser.add_argument(
        "-o", "--output",
        help="Output log file (default: date_extraction_log.txt in the target directory)",
        type=str
    )
    
    parser.add_argument(
        "-e", "--extensions",
        help="Comma-separated list of file extensions to process (e.g., 'jpg,png,pdf')",
        type=str
    )
    
    args = parser.parse_args()
    
    # Verify directory exists
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        sys.exit(1)
    
    print(f"Processing directory: {args.directory}")
    print(f"Using model: {args.model}")
    print(f"Recursive mode: {'Yes' if args.recursive else 'No'}")
    print(f"Dry run mode: {'Yes' if args.dry_run else 'No'}")
    
    if args.dry_run:
        # Override the rename function to just print instead of renaming
        import date_extractor
        original_rename = date_extractor.rename_file_with_date
        
        def dry_run_rename(file_path, date_str):
            if not date_str:
                return file_path
                
            directory = os.path.dirname(file_path)
            filename = os.path.basename(file_path)
            name_parts = os.path.splitext(filename)
            base_name = name_parts[0]
            extension = name_parts[1]
            new_filename = f"{date_str}.{base_name}{extension}"
            print(f"Would rename: {filename} -> {new_filename}")
            return file_path
            
        date_extractor.rename_file_with_date = dry_run_rename
    
    # Process the directory
    await process_directory(args.directory, args.prompt, args.model, args.recursive)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
