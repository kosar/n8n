# Date Extractor for Images

This tool automatically extracts dates from images and PDFs using Ollama's vision model capability, then renames files to include the date at the beginning of the filename.

## Features

- Extracts dates from images using AI vision model
- Converts PDFs to images for processing
- Renames files with extracted dates using format MM.DD.YY
- Interactive directory selection
- Recursive processing option for subdirectories
- Detailed logging of all operations
- Progress tracking with rich UI
- Multiple interfaces:
  - Interactive console interface
  - Command-line interface for automation
  - Graphical user interface for ease of use

## Requirements

- Python 3.6+
- Ollama installed and running with a vision model (llama3.2-vision recommended)
- Required Python packages (see installation)

## Installation

### Easy Installation

Run the installer script:

```bash
python install_date_extractor.py
```

This will:
1. Check Python version
2. Install required packages
3. Check for Ollama installation
4. Check for vision models
5. Create launch scripts

### Manual Installation

1. Install required Python packages:

```bash
pip install -r requirements.txt
```

2. Make sure Ollama is installed and running:

```bash
# Install Ollama from https://ollama.com/download
# Pull the vision model
ollama pull llama3.2-vision
```

## Usage

### Interactive Console Interface

```bash
python date_extractor.py
```

### Command-Line Interface

For batch processing or automation:

```bash
# Process current directory
python cli_date_extractor.py

# Process specific directory
python cli_date_extractor.py /path/to/images

# Process directory and subdirectories
python cli_date_extractor.py /path/to/images -r

# Dry run (don't rename files, just show what would happen)
python cli_date_extractor.py /path/to/images --dry-run

# Use different model or prompt
python cli_date_extractor.py /path/to/images -m llava -p "Find the date in this image"

# Show all options
python cli_date_extractor.py --help
```

### Graphical User Interface

```bash
python gui_date_extractor.py
```

### Follow the prompts:
   - Select directory containing images
   - Choose whether to process subdirectories
   - Customize prompt if needed
   - Select vision model
   - Confirm processing

### Check the results:
   - Files will be renamed with date prefixes
   - A log file will be created with details of all operations
   - Original PDFs are preserved, converted images are given date prefixes

## Tips

- For better date extraction, you can customize the prompt to focus on specific types of dates
- If llama3.2-vision is not available, the tool will suggest alternative models
- The log file contains detailed information about each processed file
- For PDFs, the extracted images are saved in a new folder named after the PDF
- Use the dry-run option when processing important files to preview changes before renaming
- For processing large batches of files, the command-line interface is more efficient

## Troubleshooting

If you encounter issues:

1. Make sure Ollama is running: `ollama serve`
2. Verify you have a vision model installed: `ollama list`
3. Check the log file for specific errors
4. Try with a different model if available
5. For PDFs, ensure you have Poppler installed (required by pdf2image)
