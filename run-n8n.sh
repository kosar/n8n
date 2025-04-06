#!/bin/bash

# Add debug mode flag at the top with other variables
DEBUG_MODE=false
SILENT_MODE=false

RECOMMENDED_NODE_VERSION="18.17.0"
ACCEPTABLE_VERSIONS=("18" "20" "22")
OLLAMA_API_URL="http://localhost:11434/api"
N8N_PORT=5678
N8N_URL="http://localhost:$N8N_PORT"
N8N_DIR=$(dirname "$0")
LOG_FILE="$N8N_DIR/run-n8n-debug.log" # Define log file path

# Global variables to store system status
STATUS_NODEJS=""
STATUS_NVM=""
STATUS_OLLAMA=""
STATUS_OLLAMA_RUNNING=""
STATUS_OLLAMA_MODELS=""
STATUS_OLLAMA_MODEL_LIST=()  # Array to store actual model names
STATUS_CHROME=""
STATUS_CURL=""
STATUS_JQ=""
STATUS_INTERNET=""
STATUS_LAST_SCAN=""
STATUS_N8N=""  # Added to track n8n version
STATUS_N8N_UPDATE=""  # Added to track update availability
N8N_RUNNING=false
N8N_PID=""

# Add n8n installation status tracking
N8N_DATA_DIR="$HOME/.n8n"
STATUS_N8N_INSTALLED=false

# Add these environment variables at the top of the file, after the existing variables
export N8N_ENFORCE_SETTINGS_FILE_PERMISSIONS=true
export N8N_RUNNERS_ENABLED=true
export N8N_USER_MANAGEMENT_DISABLED=true
export N8N_BASIC_AUTH_ACTIVE=false
export N8N_DISABLE_PRODUCTION_MAIN_PROCESS=true

# Function to find and source nvm
load_nvm() {
  # Common nvm locations to try
  NVM_LOCATIONS=(
    "$HOME/.nvm/nvm.sh"
    "/usr/local/opt/nvm/nvm.sh"
    "/opt/homebrew/opt/nvm/nvm.sh"
    "$NVM_DIR/nvm.sh"
  )
  
  for location in "${NVM_LOCATIONS[@]}"; do
    if [ -f "$location" ]; then
      echo "Found nvm at $location"
      source "$location"
      return 0
    fi
  done
  
  # Try to load from profile files if nvm is not found in common locations
  if command -v nvm &>/dev/null; then
    echo "nvm is available in PATH"
    return 0
  fi
  
  for profile in "$HOME/.bash_profile" "$HOME/.bashrc" "$HOME/.zshrc"; do
    if [ -f "$profile" ]; then
      echo "Attempting to load nvm from $profile"
      source "$profile" &>/dev/null
      if command -v nvm &>/dev/null; then
        echo "Successfully loaded nvm from profile"
        return 0
      fi
    fi
  done
  
  return 1
}

# Function to scan system for all required components
system_status_scan() {
  echo "🔍 Scanning system components..."
  
  # Check internet connectivity
  if ping -c 1 google.com &>/dev/null || ping -c 1 github.com &>/dev/null; then
    STATUS_INTERNET="✅ Connected"
  else
    STATUS_INTERNET="❌ Not connected or limited"
  fi

  # Check curl
  if command -v curl &>/dev/null; then
    local curl_version=$(curl --version | head -n 1 | cut -d ' ' -f 2)
    STATUS_CURL="✅ Installed ($curl_version)"
  else
    STATUS_CURL="❌ Not installed"
  fi

  # Check jq
  if command -v jq &>/dev/null; then
    local jq_version=$(jq --version 2>&1 | cut -d '-' -f 2)
    STATUS_JQ="✅ Installed ($jq_version)"
  else
    STATUS_JQ="⚠️ Not installed (optional)"
  fi

  # Check Node.js
  if command -v node &>/dev/null; then
    local node_version=$(node -v 2>/dev/null | sed 's/v//')
    if is_supported_version "$node_version"; then
      STATUS_NODEJS="✅ v$node_version (compatible)"
    else
      STATUS_NODEJS="⚠️ v$node_version (not compatible)"
    fi
  else
    STATUS_NODEJS="❌ Not installed"
  fi

  # Check nvm
  if command -v nvm &>/dev/null || [ -f "$HOME/.nvm/nvm.sh" ]; then
    STATUS_NVM="✅ Installed"
  else
    STATUS_NVM="❌ Not installed"
  fi

  # Check Chrome
  if check_chrome_installed; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # macOS - try to get version more precisely
      local chrome_version=$(/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --version 2>/dev/null | cut -d ' ' -f 3)
      STATUS_CHROME="✅ Installed ($chrome_version)"
    else
      STATUS_CHROME="✅ Installed"
    fi
  else
    STATUS_CHROME="⚠️ Not installed (using default browser)"
  fi

  # Check Ollama installed
  if command -v ollama &>/dev/null; then
    local ollama_version=$(ollama --version 2>&1 | head -n 1)
    STATUS_OLLAMA="✅ Installed ($ollama_version)"
    
    # Check Ollama running
    if curl -s --max-time 2 "${OLLAMA_API_URL}/tags" &>/dev/null; then
      STATUS_OLLAMA_RUNNING="✅ Running"
      
      # Check Ollama models
      local models_json=$(curl -s --max-time 2 "${OLLAMA_API_URL}/tags")
      if [ -n "$models_json" ]; then
        # Clear the existing model list
        STATUS_OLLAMA_MODEL_LIST=()
        
        if command -v jq &>/dev/null; then
          local model_count=$(echo "$models_json" | jq -r '.models | length')
          STATUS_OLLAMA_MODELS="✅ $model_count models available"
          
          # Extract model names to the array
          while IFS= read -r model_name; do
            STATUS_OLLAMA_MODEL_LIST+=("$model_name")
          done < <(echo "$models_json" | jq -r '.models[].name')
        else
          STATUS_OLLAMA_MODELS="✅ Models available"
          # Basic extraction without jq
          while IFS= read -r line; do
            if [[ "$line" =~ \"name\":\"([^\"]+)\" ]]; then
              STATUS_OLLAMA_MODEL_LIST+=("${BASH_REMATCH[1]}")
            fi
          done < <(echo "$models_json" | grep -o '"name":"[^"]*"')
        fi
      else
        STATUS_OLLAMA_MODELS="⚠️ No models found"
        STATUS_OLLAMA_MODEL_LIST=()
      fi
    else
      STATUS_OLLAMA_RUNNING="❌ Not running"
      STATUS_OLLAMA_MODELS="❓ Unknown (Ollama not running)"
      STATUS_OLLAMA_MODEL_LIST=()
    fi
  else
    STATUS_OLLAMA="❌ Not installed"
    STATUS_OLLAMA_RUNNING="❓ N/A"
    STATUS_OLLAMA_MODELS="❓ N/A"
    STATUS_OLLAMA_MODEL_LIST=()
  fi

  # Check n8n version
  if command -v n8n &>/dev/null; then
    local installed_version=$(n8n --version 2>/dev/null | sed 's/[^0-9\.]//g')
    STATUS_N8N="✅ v$installed_version installed"
    STATUS_N8N_UPDATE="unknown"
    
    # Try to check the latest version
    if command -v npm &>/dev/null && [ -n "$STATUS_INTERNET" ] && [[ "$STATUS_INTERNET" == *"Connected"* ]]; then
      local latest_version=$(npm view n8n version 2>/dev/null)
      if [ -n "$latest_version" ]; then
        if [ "$installed_version" != "$latest_version" ]; then
          STATUS_N8N="⚠️ v$installed_version installed"
          STATUS_N8N_UPDATE="v$latest_version available"
        else
          STATUS_N8N="✅ v$installed_version installed (up-to-date)"
          STATUS_N8N_UPDATE="current"
        fi
      fi
    fi
  else
    STATUS_N8N="❌ Not installed"
    STATUS_N8N_UPDATE="not installed"
  fi

  # Update last scan time
  STATUS_LAST_SCAN=$(date '+%Y-%m-%d %H:%M:%S')
  
  echo "✅ System scan complete!"
}

# Function to display system status in compact dashboard format
display_system_status() {
  if [ "$DEBUG_MODE" = true ] || [ "$SILENT_MODE" = true ]; then
    echo "========================================================"
    echo "               n8n System Status Dashboard               "
    echo "========================================================"
    echo ""
    echo "SYSTEM COMPONENTS:"
    printf "  %-15s %s\n" "Node.js:" "$STATUS_NODEJS"
    printf "  %-15s %s\n" "nvm:" "$STATUS_NVM"
    printf "  %-15s %s\n" "Chrome:" "$STATUS_CHROME"
    printf "  %-15s %s\n" "curl:" "$STATUS_CURL"
    printf "  %-15s %s\n" "jq:" "$STATUS_JQ"
    printf "  %-15s %s\n" "Internet:" "$STATUS_INTERNET"
    
    echo ""
    echo "N8N STATUS:"
    printf "  %-15s %s\n" "n8n:" "$STATUS_N8N"
    if [[ "$STATUS_N8N_UPDATE" != "current" && "$STATUS_N8N_UPDATE" != "unknown" && "$STATUS_N8N_UPDATE" != "not installed" ]]; then
      printf "  %-15s %s\n" "Update:" "$STATUS_N8N_UPDATE"
    fi
    
    echo ""
    echo "AI ASSISTANT STATUS:"
    printf "  %-15s %s\n" "Ollama:" "$STATUS_OLLAMA"
    printf "  %-15s %s\n" "Ollama Server:" "$STATUS_OLLAMA_RUNNING"
    printf "  %-15s %s\n" "Ollama URL:" "http://localhost:11434"
    
    # Test Ollama endpoint explicitly
    echo "  Testing Ollama endpoint..."
    if curl -s --max-time 2 "${OLLAMA_API_URL}/tags" &>/dev/null; then
      echo "  ✅ Ollama API is responding"
    else
      echo "  ❌ Ollama API is not responding"
    fi
    
    printf "  %-15s %s\n" "Ollama Models:" "$STATUS_OLLAMA_MODELS"
    
    # Display model list if any models are available
    if [ ${#STATUS_OLLAMA_MODEL_LIST[@]} -gt 0 ]; then
      echo ""
      echo "  Available Ollama models:"
      for model in "${STATUS_OLLAMA_MODEL_LIST[@]}"; do
        printf "    • %s\n" "$model"
      done
    fi
    
    echo ""
    echo "TIMESTAMP:"
    printf "  %-15s %s\n" "Last scan:" "$STATUS_LAST_SCAN"
    echo ""
    echo "========================================================"
  else
    clear
    echo "========================================================"
    echo "               n8n System Status Dashboard               "
    echo "========================================================"
    echo ""
    echo "SYSTEM COMPONENTS:"
    printf "  %-15s %s\n" "Node.js:" "$STATUS_NODEJS"
    printf "  %-15s %s\n" "nvm:" "$STATUS_NVM"
    printf "  %-15s %s\n" "Chrome:" "$STATUS_CHROME"
    printf "  %-15s %s\n" "curl:" "$STATUS_CURL"
    printf "  %-15s %s\n" "jq:" "$STATUS_JQ"
    printf "  %-15s %s\n" "Internet:" "$STATUS_INTERNET"
    
    echo ""
    echo "N8N STATUS:"
    printf "  %-15s %s\n" "n8n:" "$STATUS_N8N"
    if [[ "$STATUS_N8N_UPDATE" != "current" && "$STATUS_N8N_UPDATE" != "unknown" && "$STATUS_N8N_UPDATE" != "not installed" ]]; then
      printf "  %-15s %s\n" "Update:" "$STATUS_N8N_UPDATE"
    fi
    
    echo ""
    echo "AI ASSISTANT STATUS:"
    printf "  %-15s %s\n" "Ollama:" "$STATUS_OLLAMA"
    printf "  %-15s %s\n" "Ollama Server:" "$STATUS_OLLAMA_RUNNING"
    printf "  %-15s %s\n" "Ollama URL:" "http://localhost:11434"
    
    # Test Ollama endpoint explicitly
    echo "  Testing Ollama endpoint..."
    if curl -s --max-time 2 "${OLLAMA_API_URL}/tags" &>/dev/null; then
      echo "  ✅ Ollama API is responding"
    else
      echo "  ❌ Ollama API is not responding"
    fi
    
    printf "  %-15s %s\n" "Ollama Models:" "$STATUS_OLLAMA_MODELS"
    
    # Display model list if any models are available
    if [ ${#STATUS_OLLAMA_MODEL_LIST[@]} -gt 0 ]; then
      echo ""
      echo "  Available Ollama models:"
      local count=0
      for model in "${STATUS_OLLAMA_MODEL_LIST[@]}"; do
        printf "    • %s\n" "$model"
        ((count++))
        # After showing 10 models, summarize the rest
        if [ $count -eq 10 ] && [ ${#STATUS_OLLAMA_MODEL_LIST[@]} -gt 10 ]; then
          remaining=$((${#STATUS_OLLAMA_MODEL_LIST[@]} - 10))
          printf "    • ... and %d more model(s)\n" "$remaining"
          break
        fi
      done
    fi
    
    echo ""
    echo "TIMESTAMP:"
    printf "  %-15s %s\n" "Last scan:" "$STATUS_LAST_SCAN"
    echo ""
    echo "========================================================"
    echo ""
    echo "Press Enter to return to menu..."
    read -r
    show_interactive_menu
  fi
}

# Function to check if version is supported
is_supported_version() {
  local version=$1
  local major_version=$(echo $version | cut -d. -f1)
  
  for acceptable in "${ACCEPTABLE_VERSIONS[@]}"; do
    if [ "$major_version" = "$acceptable" ]; then
      return 0
    fi
  done
  return 1
}

# Function to check if Ollama is installed
check_ollama_installed() {
  if command -v ollama &>/dev/null; then
    echo "✅ Ollama is installed"
    return 0
  else
    echo "❌ Ollama is not installed"
    echo "To install Ollama, visit: https://ollama.ai/"
    return 1
  fi
}

# Function to check if Ollama server is running
check_ollama_running() {
  if curl -s --max-time 2 "${OLLAMA_API_URL}/tags" &>/dev/null; then
    echo "✅ Ollama server is running"
    return 0
  else
    echo "❌ Ollama server is not running"
    echo "Start Ollama with: ollama serve"
    return 1
  fi
}

# Function to list available Ollama models
list_ollama_models() {
  echo "📋 Checking available Ollama models..."
  local models_json=$(curl -s --max-time 5 "${OLLAMA_API_URL}/tags")
  
  if [ $? -ne 0 ] || [ -z "$models_json" ]; then
    echo "❌ Failed to retrieve models from Ollama"
    return 1
  fi
  
  # Check if we have jq for proper JSON parsing
  if command -v jq &>/dev/null; then
    echo "Available models:"
    echo "$models_json" | jq -r '.models[] | "  - " + .name'
  else
    # Fallback to grep/sed for basic extraction
    echo "$models_json" | grep -o '"name":"[^"]*"' | sed 's/"name":"//g' | sed 's/"//g' | sed 's/^/  - /g'
  fi
  
  return 0
}

# Function to configure n8n for Ollama
configure_n8n_for_ollama() {
  echo "🔧 Information about local Ollama server:"
  echo "  Base URL: http://localhost:11434"
  
  # Check if any models are available
  local models_json=$(curl -s --max-time 5 "${OLLAMA_API_URL}/tags")
  
  if command -v jq &>/dev/null; then
    # Try to find a good default model, preferring llama2 or mistral
    default_model=$(echo "$models_json" | jq -r '.models[] | select(.name | contains("llama2") or contains("mistral")) | .name' | head -1)
    # If no preferred model, take the first one
    if [ -z "$default_model" ]; then
      default_model=$(echo "$models_json" | jq -r '.models[0].name')
    fi
  else
    # Simple extraction of the first model
    default_model=$(echo "$models_json" | grep -o '"name":"[^"]*"' | head -1 | sed 's/"name":"//g' | sed 's/"//g')
  fi
  
  if [ -n "$default_model" ]; then
    echo "🤖 Recommended model: $default_model"
  fi
  
  echo ""
  echo "ℹ️  n8n will open in your browser shortly. To configure Ollama:"
  echo "  1. Go to Settings > AI Assistants"
  echo "  2. Select 'Ollama' as the provider"
  echo "  3. Set the Base URL to: http://localhost:11434"
  echo "  4. Select your preferred model from the dropdown"
  echo "  5. Save your settings"
  echo ""
}

# Function to check if Chrome is installed
check_chrome_installed() {
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if [ -d "/Applications/Google Chrome.app" ]; then
      return 0
    elif [ -d "$HOME/Applications/Google Chrome.app" ]; then
      return 0
    fi
  elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    if command -v google-chrome &>/dev/null || command -v google-chrome-stable &>/dev/null; then
      return 0
    fi
  elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]] || [[ "$OSTYPE" == "win"* ]]; then
    # Windows
    if [ -f "/c/Program Files/Google/Chrome/Application/chrome.exe" ] || [ -f "/c/Program Files (x86)/Google/Chrome/Application/chrome.exe" ]; then
      return 0
    fi
  fi
  return 1
}

# Function to open the n8n web UI in the browser
open_n8n_in_browser() {
  echo "🌐 Opening n8n web UI in browser..."
  
  # Wait a bit for n8n to start
  sleep 5
  
  if check_chrome_installed; then
    echo "✅ Found Chrome. Opening n8n in Chrome..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # macOS
      open -a "Google Chrome" "$N8N_URL"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
      # Linux
      if command -v google-chrome &>/dev/null; then
        google-chrome "$N8N_URL" &
      else
        google-chrome-stable "$N8N_URL" &
      fi
    elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]] || [[ "$OSTYPE" == "win"* ]]; then
      # Windows
      if [ -f "/c/Program Files/Google/Chrome/Application/chrome.exe" ]; then
        "/c/Program Files/Google/Chrome/Application/chrome.exe" "$N8N_URL"
      else
        "/c/Program Files (x86)/Google/Chrome/Application/chrome.exe" "$N8N_URL"
      fi
    fi
  else
    echo "⚠️ Chrome not found. Opening n8n in default browser..."
    echo "⚠️ Note: Safari may not be fully supported. Consider using Chrome for the best experience."
    
    if [[ "$OSTYPE" == "darwin"* ]]; then
      # macOS
      open "$N8N_URL"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
      # Linux
      if command -v xdg-open &>/dev/null; then
        xdg-open "$N8N_URL" &
      fi
    elif [[ "$OSTYPE" == "msys"* ]] || [[ "$OSTYPE" == "cygwin"* ]] || [[ "$OSTYPE" == "win"* ]]; then
      # Windows
      start "$N8N_URL"
    fi
  fi
}

# Function to clean package-lock.json files
clean_package_locks() {
  echo "🧹 Cleaning package-lock.json files..."
  
  # Find all package-lock.json files in the n8n directory
  local count=0
  while IFS= read -r file; do
    rm "$file"
    echo "   Removed: $file"
    ((count++))
  done < <(find "$N8N_DIR" -name "package-lock.json" -type f)
  
  echo "✅ Removed $count package-lock.json file(s)"
}

# Function to regenerate package-lock.json files
regenerate_package_locks() {
  echo "🔄 Regenerating package-lock.json files..."
  
  # Count package.json files (locations where we need to run npm install)
  local dirs=()
  while IFS= read -r dir; do
    dirs+=("$(dirname "$dir")")
  done < <(find "$N8N_DIR" -name "package.json" -type f)
  
  # Make unique directories
  dirs=($(echo "${dirs[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))
  
  echo "📦 Found ${#dirs[@]} package.json location(s)"
  
  # Run npm install in each directory with a package.json
  for dir in "${dirs[@]}"; do
    echo "📦 Installing dependencies in $dir"
    (cd "$dir" && npm install --package-lock-only)
  done
  
  echo "✅ All package-lock.json files have been regenerated"
}

# Function to prepare repository for git commit
git_prep() {
  echo "🧹 Preparing repository for git commit..."
  
  # Remove all package-lock.json files
  echo "🧹 Cleaning package-lock.json files..."
  local lock_count=0
  while IFS= read -r file; do
    rm "$file"
    echo "   Removed: $file"
    ((lock_count++))
  done < <(find "$N8N_DIR" -name "package-lock.json" -type f)
  echo "✅ Removed $lock_count package-lock.json file(s)"
  
  # Remove all node_modules directories
  echo "🧹 Cleaning node_modules directories..."
  local node_modules_count=0
  while IFS= read -r dir; do
    rm -rf "$dir"
    echo "   Removed: $dir"
    ((node_modules_count++))
  done < <(find "$N8N_DIR" -name "node_modules" -type d)
  echo "✅ Removed $node_modules_count node_modules director(ies)"
  
  # Remove build/dist directories
  echo "🧹 Cleaning build and distribution directories..."
  local build_count=0
  for dir_pattern in "dist" "build" ".cache" "tmp" ".tmp"; do
    while IFS= read -r dir; do
      rm -rf "$dir"
      echo "   Removed: $dir"
      ((build_count++))
    done < <(find "$N8N_DIR" -name "$dir_pattern" -type d)
  done
  echo "✅ Removed $build_count build/temp director(ies)"
  
  # Remove common log files
  echo "🧹 Cleaning log files..."
  local log_count=0
  for file_pattern in "*.log" "npm-debug.log*" "yarn-debug.log*" "yarn-error.log*" "lerna-debug.log*"; do
    while IFS= read -r file; do
      rm -f "$file"
      echo "   Removed: $file"
      ((log_count++))
    done < <(find "$N8N_DIR" -name "$file_pattern" -type f)
  done
  echo "✅ Removed $log_count log file(s)"
  
  # Remove editor/OS specific files
  echo "🧹 Cleaning editor/OS specific files..."
  local editor_count=0
  for dir_pattern in ".DS_Store" "Thumbs.db" ".idea" ".vscode" "*.swp" "*.swo"; do
    while IFS= read -r file; do
      rm -rf "$file"
      echo "   Removed: $file"
      ((editor_count++))
    done < <(find "$N8N_DIR" -name "$dir_pattern")
  done
  echo "✅ Removed $editor_count editor/OS specific file(s)"
  
  # Remove n8n specific generated files
  echo "🧹 Cleaning n8n specific generated files..."
  local n8n_count=0
  for dir_pattern in ".n8n" "~/.n8n" ".env" ".env.local"; do
    if [ -e "$N8N_DIR/$dir_pattern" ]; then
      rm -rf "$N8N_DIR/$dir_pattern"
      echo "   Removed: $N8N_DIR/$dir_pattern"
      ((n8n_count++))
    fi
  done
  echo "✅ Removed $n8n_count n8n specific file(s)"
  
  # Total count of removed items
  local total=$((lock_count + node_modules_count + build_count + log_count + editor_count + n8n_count))
  echo ""
  echo "🎉 Git preparation complete! Removed $total items total."
  echo "Repository is now ready for commit."
  echo ""
  echo "Essential files preserved:"
  echo "- run-n8n.sh"
  echo "- .nvmrc"
  echo "- README.md"
  echo "- NODE_VERSION.md"
  echo "- OLLAMA_SETUP.md"
  echo "- Other source code files"
}

# Function to show help message
show_help() {
  echo "n8n Launcher Script"
  echo "------------------"
  echo ""
  echo "Usage:"
  echo "  ./run-n8n.sh [options]"
  echo ""
  echo "Options:"
  echo "  --make-it-work    Start n8n with automatic configuration"
  echo "  --debug          Run in debug mode (no menus, direct output)"
  echo "  --silent         Run in silent mode (no menus, just make it work)"
  echo "  --help           Show this help message"
  echo ""
  echo "Note: This script requires Ollama to be running. Use run-ollama.sh to manage Ollama:"
  echo "  ./run-ollama.sh start    # Start Ollama server"
  echo "  ./run-ollama.sh pull     # Download models"
  echo "  ./run-ollama.sh status   # Check Ollama status"
  echo ""
  echo "Debug Mode:"
  echo "  The --debug option runs the script without menus and displays"
  echo "  all output directly to stdout. This is useful for troubleshooting"
  echo "  and seeing exactly what the script is doing."
  echo ""
  echo "Silent Mode:"
  echo "  The --silent option runs the script without menus and follows"
  echo "  the 'just make it work' path, displaying output directly to stdout."
  echo "  This is useful for automated or non-interactive usage."
  echo ""
  echo "Example:"
  echo "  ./run-n8n.sh --debug"
  echo "  ./run-n8n.sh --silent"
  echo ""
  exit 0
}

# Function to install Ollama
install_ollama() {
  echo "📦 Installing Ollama..."
  
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    if command -v brew &>/dev/null; then
      brew install ollama
    else
      echo "❌ Homebrew not found. Installing Homebrew first..."
      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
      brew install ollama
    fi
  elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    curl -fsSL https://ollama.com/install.sh | sh
  else
    echo "❌ Unsupported operating system"
    echo "Please install Ollama manually from: https://ollama.ai/download"
    return 1
  fi
  
  # Verify installation
  if command -v ollama &>/dev/null; then
    echo "✅ Ollama installed successfully"
    return 0
  else
    echo "❌ Failed to install Ollama"
    return 1
  fi
}

# Function to start Ollama service
start_ollama_service() {
  echo "🚀 Starting Ollama service (logging to $LOG_FILE)..."
  
  # Check if Ollama is already running
  if curl -s --max-time 2 "${OLLAMA_API_URL}/tags" &>/dev/null; then
    echo "✅ Ollama service is already running"
    return 0
  fi
  
  # Start Ollama in the background, redirecting output to log file
  ollama serve >> "$LOG_FILE" 2>&1 &
  OLLAMA_PID=$!
  
  # Wait for service to start
  local max_attempts=30
  local attempt=1
  while [ $attempt -le $max_attempts ]; do
    if curl -s --max-time 2 "${OLLAMA_API_URL}/tags" &>/dev/null; then
      echo "✅ Ollama service started successfully"
      return 0
    fi
    echo "⏳ Waiting for Ollama service to start... ($attempt/$max_attempts)"
    sleep 1
    ((attempt++))
  done
  
  echo "❌ Failed to start Ollama service"
  return 1
}

# Function to download Ollama model
download_ollama_model() {
  local model_name=$1
  
  echo "📥 Downloading Ollama model: $model_name"
  
  # Check if model is already downloaded
  if curl -s --max-time 2 "${OLLAMA_API_URL}/tags" | grep -q "\"name\":\"$model_name\""; then
    echo "✅ Model $model_name is already downloaded"
    return 0
  fi
  
  # Download the model
  ollama pull "$model_name"
  
  # Verify download
  if curl -s --max-time 2 "${OLLAMA_API_URL}/tags" | grep -q "\"name\":\"$model_name\""; then
    echo "✅ Model $model_name downloaded successfully"
    return 0
  else
    echo "❌ Failed to download model $model_name"
    return 1
  fi
}

# Function to ensure Ollama is ready
ensure_ollama_ready() {
  echo "🔍 Checking Ollama availability..."
  
  # Check if Ollama is installed
  if ! command -v ollama &>/dev/null; then
    echo "❌ Ollama is not installed"
    echo "Please install Ollama first using: ./run-ollama.sh start"
    return 1
  fi
  
  # Check if Ollama server is running
  if ! curl -s --max-time 2 "${OLLAMA_API_URL}/tags" &>/dev/null; then
    echo "❌ Ollama server is not running"
    echo "Please start Ollama first using: ./run-ollama.sh start"
    return 1
  fi
  
  # Check if any models are available
  local models_json=$(curl -s --max-time 2 "${OLLAMA_API_URL}/tags")
  if [ -z "$models_json" ] || ! echo "$models_json" | grep -q "\"models\""; then
    echo "⚠️ No models found in Ollama"
    echo "Please download a model using: ./run-ollama.sh pull mistral"
    return 1
  fi
  
  echo "✅ Ollama is ready"
  return 0
}

# Inserting show_interactive_menu function definition here
show_interactive_menu() {
  # Run system scan if it hasn't been run yet
  if [ -z "$STATUS_LAST_SCAN" ]; then
    system_status_scan >> "$LOG_FILE" 2>&1 # Log scan output
  fi

  clear
  echo "╔══════════════════ n8n Launcher Menu ═══════════════════╗"
  echo "║                                                        ║"
  echo "║  1) 🚀 Start n8n - Check Node.js version and start     ║"
  echo "║  2) ✨ Just Make It Work - Auto-fix everything         ║"
  echo "║  3) 🧹 Clean package-lock.json files                   ║"
  echo "║  4) 📦 Regenerate package-lock.json files              ║"
  echo "║  5) 🧼 Prepare for Git commit - Clean generated files  ║"
  echo "║  6) 🔍 System Status Dashboard                         ║"
  echo "║  7) ℹ️  Show help - Display command line options        ║"
  echo "║  8) 🔄 Update n8n to the latest version               ║"
  echo "║  9) 🧹 Run n8n cleanup and optimization               ║"
  echo "║ 10) 🔄 Reset n8n installation (⚠️ DELETES ALL DATA)    ║"
  echo "║  0) ❌ Exit                                            ║"
  echo "║                                                        ║"
  echo "╚════════════════════════════════════════════════════════╝"

  # Show mini status summary with n8n version info
  echo ""
  echo -n "System Status: Node.js ${STATUS_NODEJS:3:30} | "

  # Show n8n version with update info if available
  if [[ "$STATUS_N8N_UPDATE" != "current" && "$STATUS_N8N_UPDATE" != "unknown" && "$STATUS_N8N_UPDATE" != "not installed" ]]; then
    echo "n8n ${STATUS_N8N:3:20} (${STATUS_N8N_UPDATE})"
  else
    echo "n8n ${STATUS_N8N:3:30}"
  fi

  echo ""
  echo -n "Enter your choice [0-10]: "
  read -r choice

  case $choice in
    1) echo "Starting n8n with Node.js version check..."; run_normal_flow ;;
    2) echo "Activating 'Just Make It Work' mode..."; just_make_it_work ;;
    3)
      echo "Cleaning package-lock.json files..."
      clean_package_locks
      echo -n "Press Enter to return to menu..."
      read -r
      show_interactive_menu
      ;;
    4)
      echo "Regenerating package-lock.json files..."
      regenerate_package_locks
      echo -n "Press Enter to return to menu..."
      read -r
      show_interactive_menu
      ;;
    5)
      echo "Preparing for Git commit..."
      git_prep
      echo -n "Press Enter to return to menu..."
      read -r
      show_interactive_menu
      ;;
    6)
      echo "Refreshing system status..."
      system_status_scan
      display_system_status
      ;;
    7)
      clear
      show_help
      echo -n "Press Enter to return to menu..."
      read -r
      show_interactive_menu
      ;;
    8)
      echo "Checking n8n version and updating..."
      check_n8n_version
      update_n8n
      echo -n "Press Enter to return to menu..."
      read -r
      show_interactive_menu
      ;;
    9)
      echo "Running n8n cleanup and optimization..."
      run_n8n_cleanup
      echo -n "Press Enter to return to menu..."
      read -r
      show_interactive_menu
      ;;
    10)
      echo "Resetting n8n installation..."
      reset_n8n_installation
      echo -n "Press Enter to return to menu..."
      read -r
      show_interactive_menu
      ;;
    0) echo "Exiting..."; exit 0 ;;
    *)
      echo "Invalid choice. Press Enter to try again..."
      read -r
      show_interactive_menu
      ;;
  esac
}

# Modify the just_make_it_work function
just_make_it_work() {
  echo "🚀 'Just make it work' mode activated!"

  # Force kill any lingering n8n processes
  echo "🛑 Stopping any existing n8n processes..."
  pkill -f "n8n start" || true
  sleep 1

  # Ensure the .n8n directory exists and is clean
  echo "⚙️ Ensuring n8n directory permissions and cleaning config..."
  rm -f "$HOME/.n8n/config"
  echo "Config file deleted status: $? (0 means success)"
  mkdir -p "$HOME/.n8n"
  chmod 700 "$HOME/.n8n"
  echo "✅ n8n directory permissions set and config cleaned."

  # Check Ollama status before starting n8n
  echo ""
  echo "🔍 Checking Ollama status..."
  check_ollama_installed
  ollama_installed=$?
    
  if [ $ollama_installed -eq 0 ]; then
    check_ollama_running
    ollama_running=$?
    
    if [ $ollama_running -eq 0 ]; then
      list_ollama_models
      configure_n8n_for_ollama
    else
      echo ""
      echo "⚠️  You'll need to start Ollama before using AI features in n8n:"
      echo "   1. Open a new terminal"
      echo "   2. Run: ollama serve"
      echo "   3. Configure n8n AI settings after launch"
    fi
  fi

  # Start n8n with control menu instead of waiting for process
  start_n8n_with_control_menu
}

# Function to start n8n with control menu
start_n8n_with_control_menu() {
  if [ "$DEBUG_MODE" = true ] || [ "$SILENT_MODE" = true ]; then
    echo "Starting n8n in ${DEBUG_MODE:+debug}${SILENT_MODE:+silent} mode..."
    echo "----------------------------------------"
    echo "Environment variables:"
    echo "N8N_AI_ENABLED=true"
    echo "N8N_AI_PROVIDER=ollama"
    echo "N8N_AI_OLLAMA_BASE_URL=http://localhost:11434"
    echo "----------------------------------------"
    
    # Start n8n with environment variables
    export N8N_AI_ENABLED=true
    export N8N_AI_PROVIDER=ollama
    export N8N_AI_OLLAMA_BASE_URL=http://localhost:11434
    
    # Start n8n in the background
    n8n start &
    N8N_PID=$!
    
    # Wait for n8n to start
    echo "Waiting for n8n to start..."
    sleep 5
    
    # Check if n8n is running
    if kill -0 $N8N_PID 2>/dev/null; then
      echo "n8n started successfully with PID: $N8N_PID"
      echo "n8n URL: http://localhost:5678"
      echo "----------------------------------------"
      echo "Press Ctrl+C to stop n8n"
      
      # Keep the script running and show status
      while true; do
        system_status_scan
        display_system_status_while_running
        sleep 5
      done
    else
      echo "Failed to start n8n"
      exit 1
    fi
  else
    # Original interactive menu version
    echo "Starting n8n with control menu..."
    echo "----------------------------------------"
    echo "Environment variables:"
    echo "N8N_AI_ENABLED=true"
    echo "N8N_AI_PROVIDER=ollama"
    echo "N8N_AI_OLLAMA_BASE_URL=http://localhost:11434"
    echo "----------------------------------------"
    
    # Start n8n with environment variables
    export N8N_AI_ENABLED=true
    export N8N_AI_PROVIDER=ollama
    export N8N_AI_OLLAMA_BASE_URL=http://localhost:11434
    
    # Start n8n in the background
    n8n start &
    N8N_PID=$!
    
    # Wait for n8n to start
    echo "Waiting for n8n to start..."
    sleep 5
    
    # Check if n8n is running
    if kill -0 $N8N_PID 2>/dev/null; then
      echo "n8n started successfully with PID: $N8N_PID"
      echo "n8n URL: http://localhost:5678"
      echo "----------------------------------------"
      echo "Press Ctrl+C to stop n8n"
      
      # Keep the script running and show status
      while true; do
        system_status_scan
        display_system_status_while_running
        sleep 5
      done
    else
      echo "Failed to start n8n"
      exit 1
    fi
  fi
}

# Function to display system status while n8n is running
display_system_status_while_running() {
  if [ "$DEBUG_MODE" = true ] || [ "$SILENT_MODE" = true ]; then
    echo "========================================================"
    echo "               n8n System Status Dashboard               "
    echo "========================================================"
    echo ""
    echo "SYSTEM COMPONENTS:"
    printf "  %-15s %s\n" "Node.js:" "$STATUS_NODEJS"
    printf "  %-15s %s\n" "nvm:" "$STATUS_NVM"
    printf "  %-15s %s\n" "Chrome:" "$STATUS_CHROME"
    printf "  %-15s %s\n" "curl:" "$STATUS_CURL"
    printf "  %-15s %s\n" "jq:" "$STATUS_JQ"
    printf "  %-15s %s\n" "Internet:" "$STATUS_INTERNET"
    
    echo ""
    echo "N8N STATUS:"
    printf "  %-15s %s\n" "n8n:" "$STATUS_N8N"
    if [[ "$STATUS_N8N_UPDATE" != "current" && "$STATUS_N8N_UPDATE" != "unknown" && "$STATUS_N8N_UPDATE" != "not installed" ]]; then
      printf "  %-15s %s\n" "Update:" "$STATUS_N8N_UPDATE"
    fi
    
    echo ""
    echo "AI ASSISTANT STATUS:"
    printf "  %-15s %s\n" "Ollama:" "$STATUS_OLLAMA"
    printf "  %-15s %s\n" "Ollama Server:" "$STATUS_OLLAMA_RUNNING"
    printf "  %-15s %s\n" "Ollama URL:" "http://localhost:11434"
    
    # Test Ollama endpoint explicitly
    echo "  Testing Ollama endpoint..."
    if curl -s --max-time 2 "${OLLAMA_API_URL}/tags" &>/dev/null; then
      echo "  ✅ Ollama API is responding"
    else
      echo "  ❌ Ollama API is not responding"
    fi
    
    printf "  %-15s %s\n" "Ollama Models:" "$STATUS_OLLAMA_MODELS"
    
    # Display model list if any models are available
    if [ ${#STATUS_OLLAMA_MODEL_LIST[@]} -gt 0 ]; then
      echo ""
      echo "  Available Ollama models:"
      for model in "${STATUS_OLLAMA_MODEL_LIST[@]}"; do
        printf "    • %s\n" "$model"
      done
    fi
    
    echo ""
    echo "N8N SERVICE:"
    printf "  %-15s %s\n" "n8n Server:" "✅ Running (PID: $N8N_PID)"
    printf "  %-15s %s\n" "n8n URL:" "$N8N_URL"
    
    echo ""
    echo "TIMESTAMP:"
    printf "  %-15s %s\n" "Last scan:" "$STATUS_LAST_SCAN"
    echo ""
    echo "========================================================"
  else
    clear
    echo "========================================================"
    echo "               n8n System Status Dashboard               "
    echo "========================================================"
    echo ""
    echo "SYSTEM COMPONENTS:"
    printf "  %-15s %s\n" "Node.js:" "$STATUS_NODEJS"
    printf "  %-15s %s\n" "nvm:" "$STATUS_NVM"
    printf "  %-15s %s\n" "Chrome:" "$STATUS_CHROME"
    printf "  %-15s %s\n" "curl:" "$STATUS_CURL"
    printf "  %-15s %s\n" "jq:" "$STATUS_JQ"
    printf "  %-15s %s\n" "Internet:" "$STATUS_INTERNET"
    
    echo ""
    echo "N8N STATUS:"
    printf "  %-15s %s\n" "n8n:" "$STATUS_N8N"
    if [[ "$STATUS_N8N_UPDATE" != "current" && "$STATUS_N8N_UPDATE" != "unknown" && "$STATUS_N8N_UPDATE" != "not installed" ]]; then
      printf "  %-15s %s\n" "Update:" "$STATUS_N8N_UPDATE"
    fi
    
    echo ""
    echo "AI ASSISTANT STATUS:"
    printf "  %-15s %s\n" "Ollama:" "$STATUS_OLLAMA"
    printf "  %-15s %s\n" "Ollama Server:" "$STATUS_OLLAMA_RUNNING"
    printf "  %-15s %s\n" "Ollama URL:" "http://localhost:11434"
    
    # Test Ollama endpoint explicitly
    echo "  Testing Ollama endpoint..."
    if curl -s --max-time 2 "${OLLAMA_API_URL}/tags" &>/dev/null; then
      echo "  ✅ Ollama API is responding"
    else
      echo "  ❌ Ollama API is not responding"
    fi
    
    printf "  %-15s %s\n" "Ollama Models:" "$STATUS_OLLAMA_MODELS"
    
    # Display model list if any models are available
    if [ ${#STATUS_OLLAMA_MODEL_LIST[@]} -gt 0 ]; then
      echo ""
      echo "  Available Ollama models:"
      for model in "${STATUS_OLLAMA_MODEL_LIST[@]}"; do
        printf "    • %s\n" "$model"
      done
    fi
    
    echo ""
    echo "N8N SERVICE:"
    printf "  %-15s %s\n" "n8n Server:" "✅ Running (PID: $N8N_PID)"
    printf "  %-15s %s\n" "n8n URL:" "$N8N_URL"
    
    echo ""
    echo "TIMESTAMP:"
    printf "  %-15s %s\n" "Last scan:" "$STATUS_LAST_SCAN"
    echo ""
    echo "========================================================"
    echo ""
    echo "Press Enter to return to n8n control panel..."
    read -r
    show_n8n_control_menu
  fi
}

# Function to gracefully stop n8n
stop_n8n() {
  echo "🛑 Stopping n8n process..."
  if [ -n "$N8N_PID" ] && ps -p $N8N_PID > /dev/null; then
    echo "   Attempting graceful shutdown via PID $N8N_PID..."
    kill $N8N_PID &>/dev/null || true
    sleep 2 # Wait for graceful shutdown
  fi

  # Force kill any remaining n8n processes just in case
  echo "   Ensuring all n8n processes are stopped..."
  pkill -f "n8n start" || true
  pkill -f "n8n worker" || true # Also kill worker processes if any

  N8N_RUNNING=false
  echo "✅ n8n stopped"
}

# Function to check latest n8n version
check_n8n_version() {
  # Get locally installed version
  local installed_version=""
  if command -v n8n &>/dev/null; then
    installed_version=$(n8n --version 2>/dev/null | sed 's/[^0-9\.]//g')
  fi
  
  # Get latest version from npm
  local latest_version=""
  if command -v npm &>/dev/null; then
    echo "Checking for latest n8n version from npm registry..."
    latest_version=$(npm view n8n version 2>/dev/null)
  fi
  
  echo "Installed version: ${installed_version:-Not installed}"
  
  if [ -n "$latest_version" ]; then
    echo "Latest version: $latest_version"
    
    if [ -n "$installed_version" ] && [ "$installed_version" != "$latest_version" ]; then
      echo "Update available: v$installed_version → v$latest_version"
      return 1  # Update available
    elif [ -n "$installed_version" ]; then
      echo "✅ n8n is up-to-date"
      return 0  # Up-to-date
    fi
  fi
  
  return 2  # Status unknown
}

# Function to update n8n
update_n8n() {
  echo "🔄 Updating n8n to the latest version..."
  
  # Check internet connectivity first
  if ! ping -c 1 google.com &>/dev/null && ! ping -c 1 github.com &>/dev/null; then
    echo "❌ No internet connection detected. Update aborted."
    return 1
  fi
  
  # Make sure Node.js is available and in a compatible version
  if ! command -v node &>/dev/null; then
    echo "❌ Node.js not found. Please install Node.js first."
    return 1
  fi
  
  local node_version=$(node -v | sed 's/v//')
  if ! is_supported_version "$node_version"; then
    echo "⚠️ Warning: Node.js $node_version may not be fully compatible with n8n."
    echo "Consider switching to version $RECOMMENDED_NODE_VERSION."
    
    # Attempt to switch Node.js version if nvm is available
    if load_nvm; then
      echo "Switching to Node.js $RECOMMENDED_NODE_VERSION..."
      if ! nvm use $RECOMMENDED_NODE_VERSION &>/dev/null; then
        echo "Installing Node.js $RECOMMENDED_NODE_VERSION..."
        nvm install $RECOMMENDED_NODE_VERSION
      fi
      nvm use $RECOMMENDED_NODE_VERSION
      node_version=$(node -v | sed 's/v//')
      echo "Now using Node.js $node_version"
    else
      echo "Continue anyway? (y/n)"
      read -r choice
      if [[ ! "$choice" =~ ^[Yy]$ ]]; then
        echo "Update aborted."
        return 1
      fi
    fi
  fi
  
  # Get current version before update
  local old_version=""
  if command -v n8n &>/dev/null; then
    old_version=$(n8n --version 2>/dev/null | sed 's/[^0-9\.]//g')
    echo "Current version: v$old_version"
  else
    echo "n8n is not currently installed"
  fi
  
  # Stop n8n if it's running
  if [ "$N8N_RUNNING" = true ]; then
    echo "Stopping n8n before update..."
    stop_n8n
  fi
  
  # Try to update globally first
  echo "Attempting to update n8n globally..."
  if ! sudo npm install -g n8n@latest; then
    echo "Global update failed, trying with sudo..."
    if ! sudo npm install -g n8n@latest; then
      echo "Global update failed, trying local update..."
      if [ -f "$N8N_DIR/package.json" ]; then
        cd "$N8N_DIR"
        if ! npm install n8n@latest; then
          echo "❌ All update attempts failed"
          return 1
        fi
      else
        echo "❌ Could not find package.json for local update"
        return 1
      fi
    fi
  fi
  
  # Verify the installation and version
  if command -v n8n &>/dev/null; then
    local new_version=$(n8n --version 2>/dev/null | sed 's/[^0-9\.]//g')
    
    if [ -n "$old_version" ] && [ "$old_version" != "$new_version" ]; then
      echo "✅ n8n has been updated from v$old_version to v$new_version"
      
      # Offer to run post-update cleanup
      echo ""
      echo "Would you like to run post-update cleanup to remove unnecessary files? (y/n)"
      read -r cleanup_choice
      
      if [[ "$cleanup_choice" =~ ^[Yy]$ ]]; then
        run_n8n_cleanup
      fi
    elif [ -n "$old_version" ] && [ "$old_version" = "$new_version" ]; then
      echo "⚠️ Update may have failed - version remains at v$old_version"
      echo "Try running: sudo npm install -g n8n@latest"
    else
      echo "✅ n8n installed successfully (v$new_version)"
    fi
    
    # Update status variables
    system_status_scan > /dev/null 2>&1
    return 0
  else
    echo "❌ Failed to update n8n"
    return 1
  fi
}

# Function to run n8n cleanup after update
run_n8n_cleanup() {
  echo "🧹 Running post-update cleanup..."
  
  # Clear npm cache
  echo "Clearing npm cache..."
  npm cache clean --force
  
  # Remove any unnecessary node_modules in n8n directories
  if [ "$update_method" = "local" ] && [ -d "$N8N_DIR/node_modules" ]; then
    echo "Optimizing node_modules..."
    (cd "$N8N_DIR" && npm prune --production)
  fi
  
  # Clean up temporary files
  echo "Removing temporary files..."
  
  # Clean npm's temporary files
  local npm_cache_dir="$HOME/.npm/_cacache"
  if [ -d "$npm_cache_dir" ]; then
    echo "Cleaning npm cache directory..."
    find "$npm_cache_dir" -type f -name "*.lock" -delete
  fi
  
  # Clear n8n's temporary directories if they exist
  local n8n_temp_dirs=(".n8n/tmp" ".n8n/cache")
  for dir in "${n8n_temp_dirs[@]}"; do
    if [ -d "$HOME/$dir" ]; then
      echo "Cleaning $HOME/$dir..."
      rm -rf "$HOME/$dir"/*
    fi
  done
  
  echo "✅ Cleanup completed"
}

# Function to check if n8n is installed
check_n8n_installation() {
  if [ -d "$N8N_DATA_DIR" ] && [ -f "$N8N_DIR/node_modules/n8n/package.json" ]; then
    STATUS_N8N_INSTALLED=true
    return 0
  else
    STATUS_N8N_INSTALLED=false
    return 1
  fi
}

# Reset n8n installation
reset_n8n_installation() {
  echo -e "\n\033[1;31m⚠️  WARNING: You are about to completely reset your n8n installation! ⚠️\033[0m"
  echo -e "\033[1;31mThis will delete all your workflows, credentials, and settings.\033[0m"
  echo -e "\033[1;31mThis action CANNOT be undone!\033[0m"
  
  read -p "Type 'RESET' to confirm deletion or anything else to cancel: " confirmation
  
  if [ "$confirmation" == "RESET" ]; then
    echo -e "\n\033[33mStopping n8n if running...\033[0m"
    # Try to stop n8n using common methods
    pkill -f "n8n" 2>/dev/null || true
    
    echo -e "\033[33mRemoving n8n data directory...\033[0m"
    if [ -d "$N8N_DATA_DIR" ]; then
      rm -rf "$N8N_DATA_DIR"
    fi
    
    # Reinstall n8n dependencies
    echo -e "\033[33mReinstalling n8n dependencies...\033[0m"
    
    # Check if we're in a valid n8n directory with a package.json
    if [ -f "$N8N_DIR/package.json" ]; then
      cd "$N8N_DIR"
      
      # Check if package-lock.json exists
      if [ -f "package-lock.json" ]; then
        echo "Found package-lock.json, using npm ci for clean install..."
        npm ci || {
          echo -e "\033[33mFailed with npm ci, trying npm install instead...\033[0m"
          npm install
        }
      else
        echo "No package-lock.json found, using npm install..."
        npm install
      fi
      
      # Verify installation
      if [ $? -eq 0 ]; then
        echo -e "\n\033[1;32m✅ n8n dependencies reinstalled successfully!\033[0m"
      else
        echo -e "\n\033[1;31m⚠️ Dependency installation had issues.\033[0m"
        echo -e "\033[33mYou may need to manually run 'npm install' in the n8n directory.\033[0m"
      fi
    else
      echo -e "\033[33mCouldn't find package.json in $N8N_DIR\033[0m"
      echo -e "\033[33mYou may need to reinstall n8n manually with npm install -g n8n\033[0m"
    fi
    
    echo -e "\n\033[1;32m✅ n8n reset successfully! You can now start fresh.\033[0m"
  else
    echo -e "\n\033[1;32mOperation canceled. Your n8n installation remains unchanged.\033[0m"
  fi
}

# Set up trap to handle Ctrl+C in a more friendly way
trap handle_interrupt INT

# Function to handle interrupt signal (Ctrl+C)
handle_interrupt() {
  echo ""
  echo "⚠️ Interrupt detected"
  
  if [ "$N8N_RUNNING" = true ]; then
    echo "Stopping n8n gracefully..."
    stop_n8n
    echo "Returning to menu..."
    show_interactive_menu
  else
    echo "Exiting script..."
    exit 0
  fi
}

# Initial setup before menu/arguments

# Initialize log file
echo "--- Script started at $(date) --- Session: $$" > "$LOG_FILE"

# Initial system scan (silent)
system_status_scan >> "$LOG_FILE" 2>&1 # Log scan output as well

# Process command line arguments
if [ "$#" -eq 0 ]; then
  show_interactive_menu
elif [ "$1" = "--make-it-work" ]; then
  echo "Starting n8n with automatic configuration..."
  system_status_scan
  ensure_ollama_ready
  start_n8n_with_control_menu
elif [ "$1" = "--debug" ]; then
  echo "Debug mode enabled. Running without menus..."
  DEBUG_MODE=true
  system_status_scan
  ensure_ollama_ready
  start_n8n_with_control_menu
elif [ "$1" = "--silent" ]; then
  echo "Silent mode enabled. Running 'just make it work' path..."
  SILENT_MODE=true
  system_status_scan
  ensure_ollama_ready
  start_n8n_with_control_menu
else
  show_help
fi



