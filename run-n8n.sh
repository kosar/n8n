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
STATUS_N8N=""  # Added to track n8n version
STATUS_N8N_UPDATE=""  # Added to track update availability
N8N_RUNNING=false
N8N_PID=""

# Add n8n installation status tracking
N8N_DATA_DIR="$HOME/.n8n"
STATUS_N8N_INSTALLED=false

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
  echo "  --update-n8n           Update n8n to the latest version"
  echo "  --cleanup              Run n8n cleanup and optimization"
  echo "  --reset-installation   Reset n8n installation (⚠️ DELETES ALL DATA)"
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
  echo "N8N STATUS:"
  printf "  %-15s %s\n" "n8n:" "$STATUS_N8N"
  if [[ "$STATUS_N8N_UPDATE" != "current" && "$STATUS_N8N_UPDATE" != "unknown" && "$STATUS_N8N_UPDATE" != "not installed" ]]; then
    printf "  %-15s %s\n" "Update:" "$STATUS_N8N_UPDATE"
  fi
  
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
  
  # Check current installation method
  local update_method="global"
  if [ -f "$N8N_DIR/package.json" ] && grep -q '"name": "n8n"' "$N8N_DIR/package.json"; then
    update_method="local"
  fi
  
  # Get current version before update
  local old_version=""
  if command -v n8n &>/dev/null; then
    old_version=$(n8n --version 2>/dev/null | sed 's/[^0-9\.]//g')
    echo "Current version: v$old_version"
  else
    echo "n8n is not currently installed"
  fi
  
  # Create temporary file to capture npm output
  local tmp_output=$(mktemp)
  
  if [ "$update_method" = "global" ]; then
    echo "Installing latest n8n version globally..."
    # Capture both stdout and stderr from npm install
    npm install -g n8n@latest 2>&1 | tee "$tmp_output"
  else
    echo "Installing latest n8n version in local directory..."
    # Capture both stdout and stderr from npm install
    (cd "$N8N_DIR" && npm install n8n@latest 2>&1 | tee "$tmp_output")
  fi
  
  # Check for EBADENGINE warnings and extract required Node.js version
  if grep -q "EBADENGINE" "$tmp_output"; then
    echo ""
    echo "⚠️ WARNING: Node.js version incompatibility detected!"
    
    # Try to extract the required Node.js version from the warning message
    local required_version=$(grep -o "required:.*node: '>=.*'" "$tmp_output" | head -1 | grep -o ">=.*'" | tr -d "'," | cut -c3-)
    
    if [ -n "$required_version" ]; then
      echo "📌 n8n components require Node.js $required_version or newer."
      echo "   You're currently using Node.js $node_version"
      
      # Ask user if they want to upgrade Node.js version
      echo ""
      echo "Would you like to upgrade Node.js to version $required_version? (y/n)"
      read -r node_upgrade_choice
      
      if [[ "$node_upgrade_choice" =~ ^[Yy]$ ]]; then
        if load_nvm; then
          echo "Installing Node.js $required_version..."
          nvm install "$required_version"
          nvm use "$required_version"
          echo "✅ Node.js upgraded to $(node -v)"
          echo "ℹ️ You should run n8n update again for best results."
        else
          echo "❌ Could not load nvm. Please upgrade Node.js manually."
        fi
      else
        echo "ℹ️ Continuing with current Node.js version. Some features may not work correctly."
      fi
    fi
  fi
  
  # Check for deprecated package warnings
  if grep -q "deprecated" "$tmp_output"; then
    echo ""
    echo "ℹ️ Some packages used by n8n are deprecated. This is normal and doesn't affect functionality."
  fi
  
  # Remove temporary file
  rm -f "$tmp_output"
  
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
    --update-n8n)
      check_n8n_version
      update_n8n
      exit 0
      ;;
    --cleanup)
      run_n8n_cleanup
      exit 0
      ;;
    --reset-installation)
      reset_n8n_installation
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
