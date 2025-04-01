#!/usr/bin/env python3
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.logging import RichHandler
import pdfplumber
import pandas as pd
from loguru import logger
import openai
from dotenv import load_dotenv
from tqdm import tqdm
from datetime import datetime
import logging
import warnings
import contextlib

# Suppress all warnings
warnings.filterwarnings('ignore')

# Configure rich console
console = Console()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="{message}",
    level="INFO",
    colorize=True
)

# Load environment variables
load_dotenv()

@contextlib.contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output."""
    with open(os.devnull, 'w') as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

def generate_output_path(input_path: str, output_path: Optional[str], overwrite: bool = False) -> str:
    """Generate output path with timestamp if needed."""
    if output_path:
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            # Add timestamp to existing filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")
    else:
        # Generate default output path with timestamp
        input_path = Path(input_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = input_path.with_name(f"{input_path.stem}_extracted_{timestamp}.txt")
    
    return str(output_path)

class PDFExtractor:
    def __init__(self, debug: bool = False):
        self.debug = debug
        if debug:
            logger.add("pdf_extractor_debug.log", level="DEBUG")
        
        # Initialize OpenAI client with Deepseek
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            logger.warning("DEEPSEEK_API_KEY not found in environment. LLM cleaning will be disabled.")
            self.use_llm = False
        else:
            self.use_llm = False  # LLM is off by default
            openai.api_key = api_key
            openai.base_url = "https://api.deepseek.com/v1"

    def extract_text(self, pdf_path: str) -> Dict[str, Union[str, List[str], List[Dict]]]:
        """Extract text, tables, and metadata from PDF."""
        results = {
            "text": [],
            "tables": [],
            "metadata": {},
            "stats": {}
        }
        
        try:
            # Configure logging for pdfplumber
            logging.getLogger('pdfplumber').setLevel(logging.ERROR)
            logging.getLogger('pdfplumber').propagate = False
            logging.getLogger('pdfplumber').addHandler(logging.NullHandler())
            
            logger.info(f"Opening PDF file: {pdf_path}")
            with pdfplumber.open(pdf_path) as pdf:
                results["metadata"] = {
                    "pages": len(pdf.pages),
                    "info": pdf.metadata
                }
                logger.info(f"PDF opened successfully. Total pages: {len(pdf.pages)}")
                
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    # Create separate tasks for text and table extraction
                    text_task = progress.add_task(
                        "[cyan]Extracting text...",
                        total=len(pdf.pages)
                    )
                    table_task = progress.add_task(
                        "[green]Extracting tables...",
                        total=len(pdf.pages)
                    )
                    
                    for page_num, page in enumerate(pdf.pages, 1):
                        # Extract text
                        text = page.extract_text(x_tolerance=3, y_tolerance=3)
                        if text:
                            results["text"].append(text)
                            if self.debug:
                                logger.debug(f"Page {page_num}: Extracted {len(text)} characters of text")
                        progress.update(text_task, advance=1, description=f"[cyan]Extracting text from page {page_num}/{len(pdf.pages)}...")
                        
                        # Extract tables
                        tables = page.extract_tables({
                            "vertical_strategy": "lines",
                            "horizontal_strategy": "lines",
                            "intersection_y_tolerance": 3,
                            "intersection_x_tolerance": 3
                        })
                        if tables:
                            for table in tables:
                                if table:  # Ensure table is not empty
                                    results["tables"].append(table)
                            if self.debug:
                                logger.debug(f"Page {page_num}: Found {len(tables)} tables")
                        progress.update(table_task, advance=1, description=f"[green]Extracting tables from page {page_num}/{len(pdf.pages)}...")
                    
                    # Final status updates
                    logger.info(f"Text extraction complete: {len(results['text'])} chunks extracted")
                    logger.info(f"Table extraction complete: {len(results['tables'])} tables found")
            
            # Calculate statistics
            results["stats"] = {
                "total_text_chunks": len(results["text"]),
                "total_tables": len(results["tables"]),
                "total_characters": sum(len(text) for text in results["text"]),
                "average_chunk_length": sum(len(text) for text in results["text"]) / len(results["text"]) if results["text"] else 0
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    def clean_content(self, content: str) -> str:
        """Clean extracted content using Deepseek LLM if available."""
        if not self.use_llm:
            return content
            
        try:
            logger.info("Cleaning content with LLM...")
            response = openai.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a text cleaning assistant. Clean the following text to be more suitable for LLM context, removing any unnecessary formatting while preserving important information and structure."},
                    {"role": "user", "content": content}
                ],
                temperature=0.1
            )
            logger.info("Content cleaning complete")
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error cleaning content with LLM: {str(e)}")
            return content

    def save_results(self, results: Dict, output_path: str):
        """Save extracted results to a text file."""
        try:
            logger.info(f"Saving results to {output_path}...")
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write metadata
                f.write("=== PDF METADATA ===\n")
                f.write(json.dumps(results["metadata"], indent=2))
                f.write("\n\n")
                
                # Write statistics
                f.write("=== EXTRACTION STATISTICS ===\n")
                f.write(json.dumps(results["stats"], indent=2))
                f.write("\n\n")
                
                # Write text content
                f.write("=== TEXT CONTENT ===\n")
                for i, text in enumerate(results["text"], 1):
                    f.write(f"\n--- Text Chunk {i} ---\n")
                    cleaned_text = self.clean_content(text)
                    f.write(cleaned_text)
                    f.write("\n")
                
                # Write tables
                if results["tables"]:
                    f.write("\n=== TABLES ===\n")
                    for i, table in enumerate(results["tables"], 1):
                        f.write(f"\n--- Table {i} ---\n")
                        df = pd.DataFrame(table)
                        f.write(df.to_string())
                        f.write("\n")
                
            logger.info(f"Results saved successfully to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise

@click.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file path (default: input_filename_extracted_TIMESTAMP.txt)')
@click.option('--debug/--no-debug', default=False, help='Enable debug logging')
@click.option('--use-llm/--no-llm', default=False, help='Enable LLM-based content cleaning')
@click.option('--overwrite/--no-overwrite', default=False, help='Overwrite existing output file instead of adding timestamp')
def main(pdf_path: str, output: Optional[str], debug: bool, use_llm: bool, overwrite: bool):
    """Extract text and tables from PDF files optimized for LLM context."""
    try:
        # Initialize extractor
        extractor = PDFExtractor(debug=debug)
        extractor.use_llm = use_llm
        
        # Generate output path
        output_path = generate_output_path(pdf_path, output, overwrite)
        
        # Process PDF
        logger.info(f"Starting PDF processing: {pdf_path}")
        results = extractor.extract_text(pdf_path)
        
        # Display summary
        console.print(Panel.fit(
            f"Extraction Summary:\n"
            f"Pages: {results['metadata']['pages']}\n"
            f"Text Chunks: {results['stats']['total_text_chunks']}\n"
            f"Tables: {results['stats']['total_tables']}\n"
            f"Total Characters: {results['stats']['total_characters']:,}\n"
            f"Output File: {output_path}",
            title="PDF Extraction Results"
        ))
        
        # Save results
        extractor.save_results(results, output_path)
        
    except Exception as e:
        logger.error(f"Failed to process PDF: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 