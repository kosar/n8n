# Node.js Version Requirements for n8n

n8n currently supports the following Node.js versions:
- v18.17.0 (recommended)
- v20.x
- v22.x

## Using the Enhanced Run Script

We've provided a `run-n8n.sh` script with several helpful features:

- Automatically detects and uses compatible Node.js versions via nvm
- Checks for Ollama integration if you want to use local AI models
- Opens n8n in Google Chrome (if available) for best compatibility
- Falls back to your default browser if Chrome isn't installed

```bash
# Make the script executable
chmod +x run-n8n.sh

# Run n8n using the script with automatic fixes
./run-n8n.sh --make-it-work
```

## Browser Compatibility

The script will:
- Launch n8n in the background
- Open Google Chrome if it's installed (recommended)
- Fall back to your default browser if Chrome isn't available
- Display a warning if using Safari, which may have compatibility issues with n8n

## Manual Node.js Version Management

### Using nvm (Node Version Manager)

If you have nvm installed:

```bash
# Install the recommended version
nvm install 18.17.0

# Use the recommended version
nvm use 18.17.0

# Then run n8n
n8n start
```

### Without nvm

1. Download and install a compatible Node.js version from [nodejs.org](https://nodejs.org/)
2. Make sure your PATH is configured to use the newly installed version
3. Verify the version with `node -v`
4. Run n8n with `n8n start`

## Integration with Ollama

This script also supports integration with Ollama for local AI models. See [OLLAMA_SETUP.md](./OLLAMA_SETUP.md) for detailed instructions.
