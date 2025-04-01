# PDF Extractor for LLM Context

A powerful command-line tool for extracting text and tables from PDF files, optimized for use as context in Large Language Models (LLMs). The tool provides intelligent extraction with optional LLM-based content cleaning using Deepseek.

## Features

- Extracts text and tables from PDF files
- Preserves document structure and metadata
- Optional LLM-based content cleaning using Deepseek
- Rich progress indicators and logging
- Comprehensive statistics and metadata output
- Debug mode for detailed logging
- Command-line interface with Click

## Installation

1. Clone this repository or download the files
2. Install the required dependencies:
```bash
pip install -r pdf_extractor_requirements.txt
```

## Configuration

### Deepseek API Key (Optional)

To use the LLM-based content cleaning feature, you need to set up your Deepseek API key:

1. Get your API key from Deepseek
2. Set it as an environment variable:
```bash
export DEEPSEEK_API_KEY='your-api-key-here'
```

Or create a `.env` file in the same directory:
```
DEEPSEEK_API_KEY=your-api-key-here
```

## Usage

Basic usage:
```bash
python pdf_extractor.py input.pdf
```

With options:
```bash
python pdf_extractor.py input.pdf --output output.txt --debug --no-llm
```

### Command Line Options

- `pdf_path`: Path to the input PDF file (required)
- `--output`, `-o`: Output file path (default: input_filename_extracted.txt)
- `--debug/--no-debug`: Enable debug logging (default: False)
- `--no-llm/--use-llm`: Disable/enable LLM-based content cleaning (default: True)

## Output Format

The extracted content is saved in a structured format:

1. PDF Metadata
   - Number of pages
   - Document information

2. Extraction Statistics
   - Total text chunks
   - Total tables
   - Total characters
   - Average chunk length

3. Text Content
   - Chunks of extracted text
   - Each chunk is cleaned by the LLM (if enabled)

4. Tables
   - Extracted tables in a readable format

## Debug Mode

When debug mode is enabled, detailed logs are saved to `pdf_extractor_debug.log`. This is useful for troubleshooting or understanding the extraction process in detail.

## Error Handling

The tool includes comprehensive error handling and will:
- Validate input file existence
- Handle PDF processing errors gracefully
- Provide clear error messages
- Log errors with full context

## Example

```bash
# Basic extraction
python pdf_extractor.py document.pdf

# With custom output and debug mode
python pdf_extractor.py document.pdf --output result.txt --debug

# Without LLM cleaning
python pdf_extractor.py document.pdf --no-llm
```

## Notes

- The tool is optimized for LLM context, so the output format prioritizes clarity and structure over human readability
- Large PDFs may take longer to process, especially with LLM cleaning enabled
- Debug mode can generate large log files for complex documents 