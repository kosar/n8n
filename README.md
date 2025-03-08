# n8n Launcher Script

This script streamlines running n8n with the correct Node.js version, manages dependencies, and configures Ollama for AI capabilities. It provides an organized interface for common n8n management tasks.

## Benefits

This launcher script addresses several practical challenges when running n8n locally:

- **Environment Management**: Ensures consistent Node.js version compliance across different sessions and systems
- **Dependency Handling**: Automatically manages nvm, Node.js versions, and compatibility requirements  
- **Local AI Integration**: Streamlines the connection between n8n and Ollama for local AI model usage
- **Process Control**: Allows starting and stopping n8n without terminating the terminal session
- **Development Efficiency**: Provides utilities for cleaning generated files before commits
- **System Diagnostics**: Offers a comprehensive dashboard showing the state of all dependencies
- **Cross-Platform Functionality**: Works consistently across macOS, Linux, and Windows environments

Particularly valuable for Community Edition users and developers who need consistent, reproducible environments for local n8n operation.

## Quick Start

```bash
# Make the script executable
chmod +x run-n8n.sh

# Run the script with interactive menu
./run-n8n.sh

# OR run with automatic setup
./run-n8n.sh --make-it-work
```

## Features

- **Interactive Menu System**: User-friendly interface for all operations
- **n8n Control Panel**: Control n8n while it's running without exiting the script
- **System Status Dashboard**: Real-time scan of all dependencies and system components
- **Node.js Version Management**: Automatic switching to compatible Node.js versions
- **Ollama Integration**: Easy configuration for local AI models
- **Graceful Exit**: Stop n8n and return to menu without killing the script
- **Chrome Integration**: Opens n8n in Chrome for best compatibility
- **Git Preparation**: Clean repository for commits
- **Cross-Platform**: Works on macOS, Linux, and Windows (with bash)

## Interactive Menu

The script provides a comprehensive menu system:

```
╔══════════════════ n8n Launcher Menu ═══════════════════╗
║                                                        ║
║  1) 🚀 Start n8n - Check Node.js version and start     ║
║  2) ✨ Just Make It Work - Auto-fix everything         ║
║  3) 🧹 Clean package-lock.json files                   ║
║  4) 📦 Regenerate package-lock.json files              ║
║  5) 🧼 Prepare for Git commit - Clean generated files  ║
║  6) 🔍 System Status Dashboard                         ║
║  7) ℹ️  Show help - Display command line options        ║
║  0) ❌ Exit                                            ║
║                                                        ║
╚════════════════════════════════════════════════════════╝
```

## n8n Control Panel

When n8n is running, you get a dedicated control panel:

```
╔══════════════════ n8n Control Panel ════════════════════╗
║                                                         ║
║  n8n server is running in the background                ║
║  Web UI is available at: http://localhost:5678          ║
║                                                         ║
╚═════════════════════════════════════════════════════════╝

Options:
1) 🔄 Refresh browser
2) 🔍 View system status
0) ⏹️  Stop n8n and return to main menu
```

This lets you:
- Reopen n8n in your browser
- Check system status while running
- **Gracefully stop n8n** and return to the main menu

## System Status Dashboard

The script includes a comprehensive system scanner that checks:

- Node.js version and compatibility
- nvm installation
- Ollama installation and running status
- Available Ollama models
- Chrome browser installation
- Internet connectivity
- Other dependencies (curl, jq)

```
╔════════════ n8n System Status Dashboard ════════════╗
║                                                     ║
║  Node.js:      ✅ v18.17.0 (compatible)            ║
║  nvm:          ✅ Installed                        ║
║  Ollama:       ✅ Installed (0.1.24)               ║
║  Ollama Server: ✅ Running                          ║
║  Ollama Models: ✅ 3 models available               ║
║  Chrome:       ✅ Installed (123.0.6312.87)        ║
║  curl:         ✅ Installed (7.87.0)               ║
║  jq:           ✅ Installed (1.6)                  ║
║  Internet:     ✅ Connected                        ║
║                                                     ║
║  Last scan:    2023-04-05 15:30:45                 ║
║                                                     ║
╚═════════════════════════════════════════════════════╝
```

## Command Line Options

```bash
# Auto-fix everything and start n8n
./run-n8n.sh --make-it-work

# Clean package-lock.json files
./run-n8n.sh --clean-locks

# Regenerate package-lock.json files
./run-n8n.sh --regen-locks

# Clean up repository for git commits
./run-n8n.sh --git-prep

# Show help
./run-n8n.sh --help
```

## Browser Integration

The script automatically:
- Detects if Google Chrome is installed
- Opens n8n in Chrome for best compatibility
- Falls back to default browser with compatibility warning if Chrome isn't available

## Ollama Integration

If you have Ollama installed, the script will:
- Detect Ollama and check if it's running
- List available models
- Provide step-by-step instructions for configuring n8n to use your local Ollama models

See [OLLAMA_SETUP.md](./OLLAMA_SETUP.md) for detailed instructions.

## Node.js Version Management

The script ensures you're always using a compatible Node.js version:
- Automatically detects current Node.js version
- Checks compatibility with n8n requirements
- Uses nvm to switch to compatible version if needed
- Provides helpful documentation if manual steps are needed

See [NODE_VERSION.md](./NODE_VERSION.md) for more details.

## Requirements

- Bash shell
- curl
- Internet connection (for initial setup)
- Optional: nvm (Node Version Manager) - will be installed if missing
- Optional: Ollama - for local AI model integration
- Optional: Google Chrome - for best compatibility

## Smart Interruption Handling

The script properly handles Ctrl+C by:
- Gracefully stopping n8n if it's running
- Returning to menu rather than completely exiting the script
- Providing clear exit paths from any context