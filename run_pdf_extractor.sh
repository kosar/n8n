#!/bin/bash

# Exit on error
set -e

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Fixed venv name
VENV_NAME="pdf_extractor_venv"
VENV_PATH="./${VENV_NAME}"

# Function to clean up virtual environment
cleanup_venv() {
    if [ -d "${VENV_PATH}" ]; then
        echo "Cleaning up virtual environment..."
        rm -rf "${VENV_PATH}"
        echo "Cleanup complete"
    fi
}

# Function to ensure .gitignore includes venv pattern
ensure_gitignore() {
    if [ ! -f .gitignore ]; then
        echo "Creating .gitignore file..."
        touch .gitignore
    fi
    
    # Add venv pattern if not present
    if ! grep -q "pdf_extractor_venv" .gitignore; then
        echo "Adding venv pattern to .gitignore..."
        echo "pdf_extractor_venv" >> .gitignore
    fi
    
    # Add requirements marker pattern if not present
    if ! grep -q ".requirements_installed" .gitignore; then
        echo "Adding requirements marker pattern to .gitignore..."
        echo ".requirements_installed" >> .gitignore
    fi
}

# Parse command line arguments
CLEAN_VENV=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean-venv)
            CLEAN_VENV=true
            shift
            ;;
        *)
            break
            ;;
    esac
done

# Ensure .gitignore is properly configured
ensure_gitignore

# Check if Python 3 is installed
if ! command_exists python3; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check if venv module is available
if ! python3 -c "import venv" 2>/dev/null; then
    echo "Error: Python venv module is not installed"
    echo "Please install python3-venv package"
    exit 1
fi

# Clean existing venv if requested
if [ "$CLEAN_VENV" = true ]; then
    cleanup_venv
fi

# Create new venv if it doesn't exist
if [ ! -d "${VENV_PATH}" ]; then
    echo "Creating virtual environment: ${VENV_NAME}"
    python3 -m venv "${VENV_PATH}"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "${VENV_PATH}/bin/activate"

# Verify activation
if [ -z "${VIRTUAL_ENV}" ]; then
    echo "Error: Failed to activate virtual environment"
    exit 1
fi

echo "Virtual environment activated successfully"

# Only upgrade pip and install requirements if venv is new or --clean-venv was used
if [ "$CLEAN_VENV" = true ] || [ ! -f "${VENV_PATH}/.requirements_installed" ]; then
    echo "Upgrading pip..."
    python -m pip install --upgrade pip

    echo "Installing requirements..."
    pip install -r pdf_extractor_requirements.txt
    
    # Mark requirements as installed
    touch "${VENV_PATH}/.requirements_installed"
else
    echo "Using existing virtual environment with installed requirements"
fi

# Check for Deepseek API key
if [ -z "${DEEPSEEK_API_KEY}" ]; then
    echo "Note: DEEPSEEK_API_KEY is not set"
    echo "LLM-based content cleaning is disabled by default"
    echo "To enable it, set the environment variable and use --use-llm:"
    echo "export DEEPSEEK_API_KEY='your-api-key-here'"
fi

# Check if PDF file is provided
if [ $# -eq 0 ]; then
    echo "Error: No PDF file specified"
    echo "Usage: $0 [--clean-venv] <pdf_file> [options]"
    echo "Options:"
    echo "  --clean-venv          Clean and recreate the virtual environment"
    echo "  --output, -o <file>    Output file path (default: input_filename_extracted_TIMESTAMP.txt)"
    echo "  --overwrite           Overwrite existing output file instead of adding timestamp"
    echo "  --debug               Enable debug logging"
    echo "  --use-llm             Enable LLM-based content cleaning (requires DEEPSEEK_API_KEY)"
    deactivate
    exit 1
fi

# Run the PDF extractor with all provided arguments
echo "Running PDF extractor..."
python pdf_extractor.py "$@"

# Deactivate virtual environment
echo "Deactivating virtual environment..."
deactivate

echo "Virtual environment preserved: ${VENV_NAME}"
echo "To reuse this environment, activate it with:"
echo "source ${VENV_PATH}/bin/activate" 