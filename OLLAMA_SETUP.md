# Using Ollama with n8n

This guide explains how to use Ollama's local AI models with n8n.

## Prerequisites

1. Install Ollama from [ollama.ai](https://ollama.ai/)
2. Start the Ollama service: `ollama serve`
3. Download at least one model: `ollama pull llama2` or another model of your choice
4. Google Chrome is recommended for the best compatibility with n8n's UI

## Automatic Setup

The `run-n8n.sh` script will automatically:
- Check if Ollama is installed and running
- List available models
- Launch n8n and open it in Google Chrome (if installed)
- Provide instructions for connecting n8n to Ollama

Simply run:
```bash
./run-n8n.sh --make-it-work
```

## Browser Compatibility

The script will:
- Use Google Chrome if it's installed (recommended for best compatibility)
- Fall back to your default browser if Chrome isn't available
- Display a warning if using Safari, which may have compatibility issues with n8n

## Manual Setup in n8n UI

To configure n8n to use Ollama:

1. Start n8n
2. Navigate to Settings > AI Assistants
3. Choose "Ollama" as the AI provider
4. Set the Base URL to: `http://localhost:11434`
5. Select one of your downloaded models from the dropdown
6. Save the settings

## Troubleshooting

If n8n cannot connect to Ollama:

1. Verify Ollama is running: `ps aux | grep ollama`
2. Check Ollama's API: `curl http://localhost:11434/api/tags`
3. Ensure your firewall allows connections to port 11434
4. Restart Ollama: `ollama serve`

## Managing Ollama Models

- List available models: `ollama list`
- Download a new model: `ollama pull modelname`
- Remove a model: `ollama rm modelname`
- Get model information: `ollama show modelname`

For more details, visit the [Ollama documentation](https://github.com/ollama/ollama/blob/main/README.md).
