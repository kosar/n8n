#!/bin/bash

RECOMMENDED_NODE_VERSION="18.17.0"
ACCEPTABLE_VERSIONS=("18" "20" "22")
OLLAMA_API_URL="http://localhost:11434/api"
N8N_PORT=5678
N8N_URL="http://localhost:$N8N_PORT"
N8N_DIR=$(dirname "$0")

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
N8N_RUNNING=false
N8N_PID=""

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

  # Update last scan time
  STATUS_LAST_SCAN=$(date '+%Y-%m-%d %H:%M:%S')
  
  echo "✅ System scan complete!"
}

# Function to display system status in compact dashboard format
display_system_status() {
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
  echo "AI ASSISTANT STATUS:"
  printf "  %-15s %s\n" "Ollama:" "$STATUS_OLLAMA"
  printf "  %-15s %s\n" "Ollama Server:" "$STATUS_OLLAMA_RUNNING"
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

# Function to show help
show_help() {
  echo "Usage: ./run-n8n.sh [options]"
  echo ""
  echo "Options:"
  echo "  --make-it-work         Auto-fix Node.js version and start n8n"
  echo "  --clean-locks          Remove all package-lock.json files"
  echo "  --regen-locks          Regenerate all package-lock.json files"
  echo "  --git-prep             Thoroughly clean up all generated files for git commit"
  echo "  --help                 Show this help message"
}

# Just make it work mode
just_make_it_work() {
  echo "🚀 'Just make it work' mode activated!"
  
  # Handle Node.js version
  if ! load_nvm; then
    echo "❌ Could not find or load nvm."
    echo "Installing nvm automatically..."
    
    # Try to install nvm
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash
    
    # Try to load the newly installed nvm
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    
    if ! command -v nvm &>/dev/null; then
      echo "❌ Failed to install nvm automatically."
      echo "Please install nvm manually: https://github.com/nvm-sh/nvm#installing-and-updating"
      exit 1
    fi
  fi
  
  # Now that nvm is loaded, install and use the recommended version
  echo "📦 Installing Node.js $RECOMMENDED_NODE_VERSION..."
  nvm install $RECOMMENDED_NODE_VERSION
  
  echo "🔄 Switching to Node.js $RECOMMENDED_NODE_VERSION..."
  nvm use $RECOMMENDED_NODE_VERSION
  
  # Verify the version
  NODE_VERSION=$(node -v)
  echo "✅ Now using Node.js $NODE_VERSION"
  
  # Check Ollama status
  echo ""
  echo "🔍 Checking Ollama status..."
  check_ollama_installed
  ollama_installed=$?
  
  if [ $ollama_installed -eq 0 ]; then
    check_ollama_running
    ollama_running=$?
    
    if (ollama_running -eq 0); then
      list_ollama_models
      configure_n8n_for_ollama
    else
      echo ""
      echo "⚠️  You'll need to start Ollama before using AI features in n8n:"
      echo "   1. Open a new terminal"
      echo "   2. Run: ollama serve"
      echo "   3. Return to n8n to configure AI settings"
    fi
  fi
  
  # Start n8n with control menu instead of waiting for process
  start_n8n_with_control_menu
}

# Function to show an interactive menu (compact version)
show_interactive_menu() {
  # Run system scan if it hasn't been run yet
  if [ -z "$STATUS_LAST_SCAN" ]; then
    system_status_scan > /dev/null 2>&1
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
  echo "║  0) ❌ Exit                                            ║"
  echo "║                                                        ║"
  echo "╚════════════════════════════════════════════════════════╝"
  
  # Show mini status summary
  echo ""
  echo "System Status: Node.js ${STATUS_NODEJS:3:30} | Ollama ${STATUS_OLLAMA_RUNNING:3:10}"
  echo ""
  echo -n "Enter your choice [0-7]: "
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
    0) echo "Exiting..."; exit 0 ;;
    *)
      echo "Invalid choice. Press Enter to try again..."
      read -r
      show_interactive_menu
      ;;
  esac
}

# Function to run the normal execution flow with automatic Node.js version fix
run_normal_flow() {
  # Check current Node.js version
  CURRENT_VERSION=$(node -v 2>/dev/null | sed 's/v//')

  # Check if current version is supported
  if ! is_supported_version "$CURRENT_VERSION"; then
    echo "Node.js $CURRENT_VERSION is not supported. Switching to compatible version..."
    
    # Try to load nvm
    if load_nvm; then
      # Use the recommended Node.js version
      echo "Using nvm to switch to Node.js $RECOMMENDED_NODE_VERSION..."
      
      # Try to use existing installation first
      if ! nvm use $RECOMMENDED_NODE_VERSION &>/dev/null; then
        # If not installed, install it
        echo "Installing Node.js $RECOMMENDED_NODE_VERSION..."
        nvm install $RECOMMENDED_NODE_VERSION
        nvm use $RECOMMENDED_NODE_VERSION
      fi
      
      # Verify the version
      NODE_VERSION=$(node -v)
      echo "Now using Node.js $NODE_VERSION"
    else
      echo "Could not find nvm. Please select 'Just Make It Work' option instead."
      echo "Press Enter to return to menu..."
      read -r
      show_interactive_menu
      return
    fi
  else
    echo "Using Node.js $CURRENT_VERSION"
  fi

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

# Function to start n8n and show control menu
start_n8n_with_control_menu() {
  # Start n8n in the background
  echo "🚀 Starting n8n..."
  n8n start &
  N8N_PID=$!
  N8N_RUNNING=true
  
  # Wait a moment for n8n to start
  sleep 2
  
  # Open in browser
  open_n8n_in_browser
  
  # Show control menu
  show_n8n_control_menu
}

# Function to show the n8n control menu while n8n is running
show_n8n_control_menu() {
  clear
  echo "╔══════════════════ n8n Control Panel ════════════════════╗"
  echo "║                                                         ║"
  echo "║  n8n server is running in the background                ║"
  echo "║  Web UI is available at: $N8N_URL                  ║"
  echo "║                                                         ║"
  echo "╚═════════════════════════════════════════════════════════╝"
  echo ""
  echo "Options:"
  echo "1) 🔄 Refresh browser"
  echo "2) 🔍 View system status"
  echo "0) ⏹️  Stop n8n and return to main menu"
  echo ""
  echo -n "Enter your choice [0-2]: "
  read -r control_choice

  case $control_choice in
    1)
      echo "Reopening n8n in browser..."
      open_n8n_in_browser
      show_n8n_control_menu
      ;;
    2)
      echo "Refreshing system status..."
      system_status_scan
      display_system_status_while_running
      ;;
    0)
      echo "Stopping n8n server..."
      stop_n8n
      echo "Returning to main menu..."
      show_interactive_menu
      ;;
    *)
      echo "Invalid choice. Press Enter to try again..."
      read -r
      show_n8n_control_menu
      ;;
  esac
}

# Function to display system status while n8n is running
display_system_status_while_running() {
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
  echo "AI ASSISTANT STATUS:"
  printf "  %-15s %s\n" "Ollama:" "$STATUS_OLLAMA"
  printf "  %-15s %s\n" "Ollama Server:" "$STATUS_OLLAMA_RUNNING"
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
  echo "N8N SERVICE:"
  printf "  %-15s %s\n" "n8n Server:" "✅ Running (PID: $N8N_PID)"
  
  echo ""
  echo "TIMESTAMP:"
  printf "  %-15s %s\n" "Last scan:" "$STATUS_LAST_SCAN"
  echo ""
  echo "========================================================"
  echo ""
  echo "Press Enter to return to n8n control panel..."
  read -r
  show_n8n_control_menu
}

# Function to gracefully stop n8n
stop_n8n() {
  if [ -n "$N8N_PID" ] && ps -p $N8N_PID > /dev/null; then
    echo "Stopping n8n process (PID: $N8N_PID)..."
    
    # Try graceful shutdown first with n8n CLI if possible
    if command -v n8n &>/dev/null; then
      n8n stop &>/dev/null || true
    fi

    # If still running, send SIGTERM
    if ps -p $N8N_PID > /dev/null; then
      kill $N8N_PID &>/dev/null || true
      sleep 1
    fi
    
    # If still running, force kill
    if ps -p $N8N_PID > /dev/null; then
      kill -9 $N8N_PID &>/dev/null || true
    fi
    
    N8N_RUNNING=false
    echo "✅ n8n stopped"
  else
    echo "n8n is not running"
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

# Initial system scan (silent)
system_status_scan > /dev/null 2>&1

# Process command line arguments
if [ $# -eq 0 ]; then
  # No arguments provided, show interactive menu
  show_interactive_menu
else
  # Arguments provided, process them
  case "$1" in
    --make-it-work)
      just_make_it_work
      ;;
    --clean-locks)
      clean_package_locks
      exit 0
      ;;
    --regen-locks)
      regenerate_package_locks
      exit 0
      ;;
    --git-prep)
      git_prep
      exit 0
      ;;
    --help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
fi
