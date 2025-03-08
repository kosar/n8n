# n8n Launcher Script

This repository contains helper scripts for running n8n with the correct Node.js version and optional Ollama AI integration. The scripts are designed to make setup as seamless as possible across different operating systems.

## Quick Start

For the fastest and easiest setup:

```bash
# Make the script executable
chmod +x run-n8n.sh

# Run the script to see the interactive menu
./run-n8n.sh
```

## Interactive Menu

When run without arguments, the script displays a user-friendly menu with options:
- Start n8n (automatically switches to compatible Node.js version)
- Just Make It Work Mode (complete automatic setup with detailed guidance)
- Clean package-lock.json files
- Regenerate package-lock.json files
- Prepare for Git commit (clean all generated files)
- Show help
- Exit

## Features

This launcher script provides:

- **Interactive Menu**: User-friendly interface for all features
- **Node.js Version Management**: Automatically installs and uses a compatible Node.js version
- **Browser Optimization**: Opens n8n in Google Chrome for best compatibility
- **Ollama Integration**: Checks for local Ollama setup and guides you through configuration
- **Package Lock Management**: Clean and regenerate package-lock.json files as needed
- **Git Preparation**: Thoroughly clean all generated files before committing to git
- **Cross-Platform Support**: Works on macOS, Linux, and Windows (with bash shell)

## Configuration Files

- `.nvmrc`: Specifies the recommended Node.js version (18.17.0)
- `run-n8n.sh`: The main launcher script with all functionality

## Detailed Documentation

For more detailed information:

- [NODE_VERSION.md](./NODE_VERSION.md) - Details about Node.js requirements
- [OLLAMA_SETUP.md](./OLLAMA_SETUP.md) - Instructions for Ollama AI integration

## Command Line Options

If you prefer using command line arguments instead of the menu:

```bash
# Just make it work (auto-fix everything and start)
./run-n8n.sh --make-it-work

# Remove all package-lock.json files
./run-n8n.sh --clean-locks

# Regenerate package-lock.json files
./run-n8n.sh --regen-locks

# Clean all generated files for git commit
./run-n8n.sh --git-prep

# Show help
./run-n8n.sh --help
```

## Browser Compatibility

The script prioritizes Google Chrome for the n8n web interface, as it provides the best compatibility. If Chrome isn't available, it will fall back to your default browser with a warning about potential compatibility issues.

## Requirements

- Bash shell
- curl
- Internet connection (for initial setup)
- Optional: nvm (Node Version Manager) - will be installed if missing
- Optional: Ollama - for local AI model integration