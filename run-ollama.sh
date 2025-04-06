#!/bin/bash

# Ollama management script
OLLAMA_API_URL="http://localhost:11434/api"
LOG_FILE="$HOME/.ollama/ollama.log"
PID_FILE="$HOME/.ollama/ollama.pid"

# Add at the top with other global variables
declare -a STATUS_LOADED_MODELS=()

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
  # First check if we have a PID file
  if [ -f "$PID_FILE" ]; then
    local pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      # Process exists, now check if API is responding
      if curl -s --max-time 2 "${OLLAMA_API_URL}/tags" &>/dev/null; then
        echo "✅ Ollama server is running (PID: $pid)"
        return 0
      else
        # Process exists but API not responding - server might be stuck
        echo "⚠️ Ollama process exists (PID: $pid) but API not responding"
        return 2
      fi
    else
      # PID file exists but process is dead
      echo "⚠️ Ollama process not found (stale PID file)"
      rm -f "$PID_FILE"
      return 1
    fi
  else
    # No PID file, check if API is responding
    if curl -s --max-time 2 "${OLLAMA_API_URL}/tags" &>/dev/null; then
      # API is responding but no PID file - server running without our tracking
      echo "⚠️ Ollama API is responding but process not tracked"
      return 3
    else
      echo "❌ Ollama server is not running"
      return 1
    fi
  fi
}

# Function to start Ollama server
start_ollama() {
  echo "🚀 Starting Ollama server..."
  
  # First check if server is already running
  check_ollama_running
  local status=$?
  
  if [ $status -eq 0 ]; then
    echo "✅ Ollama server is already running"
    return 0
  elif [ $status -eq 2 ]; then
    echo "⚠️ Found stuck Ollama process, attempting to restart..."
    stop_ollama
    sleep 2
  elif [ $status -eq 3 ]; then
    echo "⚠️ Found untracked Ollama process, attempting to stop it..."
    pkill -f "ollama serve"
    sleep 2
  fi
  
  # Create log directory if it doesn't exist
  mkdir -p "$(dirname "$LOG_FILE")"
  
  # Start Ollama in the background
  ollama serve >> "$LOG_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  
  # Wait for server to start
  local max_attempts=30
  local attempt=1
  while [ $attempt -le $max_attempts ]; do
    if curl -s --max-time 2 "${OLLAMA_API_URL}/tags" &>/dev/null; then
      echo "✅ Ollama server started successfully (PID: $(cat "$PID_FILE"))"
      return 0
    fi
    echo "⏳ Waiting for Ollama server to start... ($attempt/$max_attempts)"
    sleep 1
    ((attempt++))
  done
  
  echo "❌ Failed to start Ollama server"
  rm -f "$PID_FILE"
  return 1
}

# Function to stop Ollama server
stop_ollama() {
  echo "🛑 Stopping Ollama server..."
  
  # First try to stop using PID file
  if [ -f "$PID_FILE" ]; then
    local pid=$(cat "$PID_FILE")
    if kill -0 "$pid" 2>/dev/null; then
      echo "   Stopping process $pid..."
      kill "$pid"
      sleep 2
      if kill -0 "$pid" 2>/dev/null; then
        echo "   Process still running, forcing termination..."
        kill -9 "$pid"
      fi
      rm -f "$PID_FILE"
    fi
  fi
  
  # Then try to find and stop any remaining Ollama processes
  local remaining_pids=$(pgrep -f "ollama serve")
  if [ -n "$remaining_pids" ]; then
    echo "   Stopping remaining Ollama processes..."
    pkill -f "ollama serve"
    sleep 2
  fi
  
  # Final check
  if curl -s --max-time 2 "${OLLAMA_API_URL}/tags" &>/dev/null; then
    echo "❌ Failed to stop Ollama server completely"
    return 1
  else
    echo "✅ Ollama server stopped"
    return 0
  fi
}

# Function to list available models
list_models() {
  echo "📋 Available Ollama models:"
  local models_json=$(curl -s --max-time 5 "${OLLAMA_API_URL}/tags")
  
  if [ $? -ne 0 ] || [ -z "$models_json" ]; then
    echo "❌ Failed to retrieve models from Ollama"
    return 1
  fi
  
  if command -v jq &>/dev/null; then
    echo "$models_json" | jq -r '.models[] | "  - " + .name'
  else
    echo "$models_json" | grep -o '"name":"[^"]*"' | sed 's/"name":"//g' | sed 's/"//g' | sed 's/^/  - /g'
  fi
}

# Function to download a model
download_model() {
  local model_name=$1
  
  if [ -z "$model_name" ]; then
    echo "❌ Please specify a model name"
    return 1
  fi
  
  echo "📥 Downloading model: $model_name"
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

# Function to test a model
test_model() {
  local model_name=$1
  
  if [ -z "$model_name" ]; then
    echo "❌ Please specify a model name"
    return 1
  fi
  
  echo "🧪 Testing model: $model_name"
  echo "Sending test prompt: 'What is 2+2?'"
  
  local response=$(curl -s --max-time 10 "${OLLAMA_API_URL}/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$model_name\", \"prompt\": \"What is 2+2?\", \"stream\": false}")
  
  if [ $? -eq 0 ] && [ -n "$response" ]; then
    echo "✅ Model test successful"
    echo "Response:"
    echo "$response" | jq -r '.response' 2>/dev/null || echo "$response"
    return 0
  else
    echo "❌ Model test failed"
    return 1
  fi
}

# Function to get system memory info (works on both Linux and macOS)
get_memory_info() {
  if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    local total_mem=$(vm_stat | awk '/free/ {print $3}' | sed 's/\.//' | awk '{printf "%.2f GB\n", $1/1024/1024/1024}')
    local used_mem=$(vm_stat | awk '/active/ {print $3}' | sed 's/\.//' | awk '{printf "%.2f GB\n", $1/1024/1024/1024}')
    echo "Total: $total_mem, Used: $used_mem"
  else
    # Linux
    local total_mem=$(free -h | grep Mem | awk '{print $2}')
    local used_mem=$(free -h | grep Mem | awk '{print $3}')
    echo "Total: $total_mem, Used: $used_mem"
  fi
}

# Function to verify model is loaded (more robust check)
verify_model_loaded() {
  local model_name=$1
  local max_attempts=5
  local attempt=1
  
  echo "🔍 Verifying model is loaded (attempt $attempt/$max_attempts)..."
  
  while [ $attempt -le $max_attempts ]; do
    # Method 1: Check metrics endpoint
    local metrics_check=$(curl -s --max-time 2 "${OLLAMA_API_URL}/metrics" | grep -o "\"model\":\"[^\"]*\"" | cut -d'"' -f4)
    if echo "$metrics_check" | grep -q "^$model_name$"; then
      echo "✅ Model verified via metrics endpoint"
      return 0
    fi
    
    # Method 2: Try a test generation
    local test_response=$(curl -s --max-time 5 "${OLLAMA_API_URL}/generate" \
      -H "Content-Type: application/json" \
      -d "{\"model\": \"$model_name\", \"prompt\": \"test\", \"stream\": false}")
    
    if [ $? -eq 0 ] && [ -n "$test_response" ]; then
      echo "✅ Model verified via test generation"
      return 0
    fi
    
    # Method 3: Check model status in tags endpoint
    local tags_check=$(curl -s --max-time 2 "${OLLAMA_API_URL}/tags" | jq -r ".models[] | select(.name==\"$model_name\") | .last_used" 2>/dev/null)
    if [ -n "$tags_check" ] && [ "$tags_check" != "null" ]; then
      echo "✅ Model verified via tags endpoint"
      return 0
    fi
    
    echo "⏳ Waiting for model to be fully loaded... (attempt $attempt/$max_attempts)"
    sleep 2
    ((attempt++))
  done
  
  return 1
}

# Function to load model into memory
load_model() {
  local model_name=$1
  echo "📥 Loading model into memory: $model_name"
  
  # First check if model exists
  if ! curl -s --max-time 2 "${OLLAMA_API_URL}/tags" | grep -q "\"name\":\"$model_name\""; then
    echo "❌ Model $model_name not found."
    echo "Would you like to download it now? (y/n)"
    read -r choice
    if [[ "$choice" =~ ^[Yy]$ ]]; then
      echo "📥 Downloading model $model_name..."
      ollama pull "$model_name"
      if [ $? -ne 0 ]; then
        echo "❌ Failed to download model $model_name"
        return 1
      fi
      echo "✅ Model downloaded successfully"
    else
      return 1
    fi
  fi
  
  # Check system resources before loading
  echo "💻 Checking system resources..."
  local mem_info=$(get_memory_info)
  echo "  Memory: $mem_info"
  
  # Try to load the model with increased timeout
  echo "🔄 Loading model into memory..."
  local response=$(curl -s --max-time 60 "${OLLAMA_API_URL}/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$model_name\", \"prompt\": \"test\", \"stream\": false}")
  
  if [ $? -eq 0 ] && [ -n "$response" ]; then
    # Verify the model is actually loaded
    if verify_model_loaded "$model_name"; then
      echo "✅ Model $model_name loaded into memory successfully"
      # Update the global status array
      if [[ ! " ${STATUS_LOADED_MODELS[@]} " =~ " ${model_name} " ]]; then
        STATUS_LOADED_MODELS+=("$model_name")
      fi
      return 0
    else
      echo "❌ Model appears to have failed loading despite successful response"
      echo "📝 Recent Ollama server logs:"
      echo "----------------------------------------"
      if [ -f "$LOG_FILE" ]; then
        tail -n 50 "$LOG_FILE" | sed 's/^/  /'
      else
        journalctl -u ollama -n 50 2>/dev/null || echo "No system logs available"
      fi
      
      # Check system resources again
      echo "💻 System Resources After Failed Load:"
      echo "  Memory: $(get_memory_info)"
      echo "  Disk Space: $(df -h / | tail -1 | awk '{print $4}')"
      
      echo "🔧 Troubleshooting steps:"
      echo "1. Check if you have enough free memory"
      echo "2. Try stopping and restarting the Ollama server"
      echo "3. Verify the model file integrity"
      echo "4. Check disk space availability"
      echo "5. Try downloading the model again"
      echo "6. Check Ollama server logs for errors"
      
      return 1
    fi
  else
    echo "❌ Failed to load model $model_name"
    
    # Get recent server logs
    echo "📝 Recent Ollama server logs:"
    echo "----------------------------------------"
    if [ -f "$LOG_FILE" ]; then
      tail -n 50 "$LOG_FILE" | sed 's/^/  /'
    else
      journalctl -u ollama -n 50 2>/dev/null || echo "No system logs available"
    fi
    
    # Check system resources
    echo "💻 System Resources:"
    echo "  Memory: $(get_memory_info)"
    echo "  Disk Space: $(df -h / | tail -1 | awk '{print $4}')"
    
    echo "🔧 Troubleshooting steps:"
    echo "1. Check if you have enough free memory"
    echo "2. Try stopping and restarting the Ollama server"
    echo "3. Verify the model file integrity"
    echo "4. Check disk space availability"
    echo "5. Try downloading the model again"
    echo "6. Check Ollama server logs for errors"
    
    return 1
  fi
}

# Function to check currently loaded models
check_loaded_models() {
  echo "🔍 Checking currently loaded models..."
  
  # Try multiple methods to get loaded models
  local loaded_models=()
  
  # Method 1: Check metrics endpoint
  local metrics_models=$(curl -s --max-time 2 "${OLLAMA_API_URL}/metrics" | grep -o "\"model\":\"[^\"]*\"" | cut -d'"' -f4)
  if [ -n "$metrics_models" ]; then
    while IFS= read -r model; do
      loaded_models+=("$model")
    done <<< "$metrics_models"
  fi
  
  # Method 2: Check tags endpoint for recently used models
  local tags_models=$(curl -s --max-time 2 "${OLLAMA_API_URL}/tags" | jq -r '.models[] | select(.last_used != null) | .name' 2>/dev/null)
  if [ -n "$tags_models" ]; then
    while IFS= read -r model; do
      # Only add if not already in array
      if [[ ! " ${loaded_models[@]} " =~ " ${model} " ]]; then
        loaded_models+=("$model")
      fi
    done <<< "$tags_models"
  fi
  
  # Method 3: Try a test generation for each model to verify it's actually loaded
  for model in "${loaded_models[@]}"; do
    local test_response=$(curl -s --max-time 5 "${OLLAMA_API_URL}/generate" \
      -H "Content-Type: application/json" \
      -d "{\"model\": \"$model\", \"prompt\": \"test\", \"stream\": false}")
    
    if [ $? -eq 0 ] && [ -n "$test_response" ]; then
      echo "  • $model"
    fi
  done
  
  # Update global status array
  STATUS_LOADED_MODELS=("${loaded_models[@]}")
  
  if [ ${#loaded_models[@]} -eq 0 ]; then
    echo "📥 Currently loaded: ℹ️ None"
  fi
  
  return 0
}

# Function to unload a model from memory
unload_model() {
  local model_name=$1
  
  if [ -z "$model_name" ]; then
    echo "❌ Please specify a model name"
    return 1
  fi
  
  echo "📤 Unloading model from memory: $model_name"
  
  # First check if model exists
  if ! curl -s --max-time 2 "${OLLAMA_API_URL}/tags" | grep -q "\"name\":\"$model_name\""; then
    echo "❌ Model $model_name not found"
    return 1
  fi
  
  # Unload the model by making a request to remove it from memory
  local response=$(curl -s --max-time 10 "${OLLAMA_API_URL}/delete" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"$model_name\"}")
  
  if [ $? -eq 0 ]; then
    echo "✅ Model $model_name unloaded from memory successfully"
    # Remove from global status array
    STATUS_LOADED_MODELS=("${STATUS_LOADED_MODELS[@]/$model_name/}")
    return 0
  else
    echo "❌ Failed to unload model $model_name"
    return 1
  fi
}

# Function to show detailed status
show_status() {
  echo "========================================================"
  echo "               Ollama Server Status                     "
  echo "========================================================"
  
  # Check installation
  check_ollama_installed
  local installed=$?
  
  # Check if running
  check_ollama_running
  local running=$?
  
  if [ $installed -eq 0 ] && [ $running -eq 0 ]; then
    echo ""
    echo "Server Information:"
    echo "  URL: http://localhost:11434"
    echo "  API: ${OLLAMA_API_URL}"
    echo "  Log File: $LOG_FILE"
    echo "  PID File: $PID_FILE"
    
    # Get server version
    local server_version=$(curl -s --max-time 2 "${OLLAMA_API_URL}/version" | jq -r '.version' 2>/dev/null)
    if [ -n "$server_version" ]; then
      echo "  Server Version: $server_version"
    fi
    
    # Get system information
    echo ""
    echo "System Information:"
    echo "  CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")"
    echo "  Memory: $(get_memory_info)"
    echo "  Disk Space: $(df -h / | awk 'NR==2 {print $4}')"
    
    echo ""
    echo "Available Models:"
    list_models
    
    # Show currently loaded models
    echo ""
    echo "Currently Loaded Models:"
    local loaded_models=$(check_loaded_models)
    if [ $? -eq 0 ]; then
      echo "$loaded_models" | while read -r model; do
        echo "  ✅ $model"
      done
    else
      echo "  ℹ️ No models currently loaded in memory"
    fi
    
    # Show recent log entries
    if [ -f "$LOG_FILE" ]; then
      echo ""
      echo "Recent Log Entries:"
      tail -n 5 "$LOG_FILE" | sed 's/^/  /'
    fi
    
    # Show model statistics
    echo ""
    echo "Model Statistics:"
    local models_json=$(curl -s --max-time 2 "${OLLAMA_API_URL}/tags")
    if [ -n "$models_json" ]; then
      local total_models=$(echo "$models_json" | jq -r '.models | length')
      echo "  Total Models: $total_models"
      
      # Show model sizes
      echo "  Model Sizes:"
      echo "$models_json" | jq -r '.models[] | "    - \(.name): \(.size / 1024 / 1024 / 1024 | round) GB"'
    fi
  fi
  
  echo "========================================================"
}

# Function to show help
show_help() {
  echo "Ollama Server Manager"
  echo "-------------------"
  echo ""
  echo "Usage:"
  echo "  ./run-ollama.sh [command]"
  echo ""
  echo "Commands:"
  echo "  start     Start the Ollama server"
  echo "  stop      Stop the Ollama server"
  echo "  status    Show server status and available models"
  echo "  list      List available models"
  echo "  pull      Download a model (requires model name)"
  echo "  test      Test a model (requires model name)"
  echo "  load      Load a model into memory (requires model name)"
  echo "  unload    Unload a model from memory (requires model name)"
  echo "  help      Show this help message"
  echo ""
  echo "Examples:"
  echo "  ./run-ollama.sh start"
  echo "  ./run-ollama.sh pull mistral"
  echo "  ./run-ollama.sh status"
  echo ""
  exit 0
}

# Function to show model selection menu
show_model_selection_menu() {
  local models_json=$(curl -s --max-time 2 "${OLLAMA_API_URL}/tags")
  if [ -n "$models_json" ]; then
    echo "Available Models:"
    echo "----------------------------------------"
    
    # Get model names and format them for display
    local model_names=($(echo "$models_json" | jq -r '.models[].name' 2>/dev/null))
    local count=1
    for model in "${model_names[@]}"; do
      printf "%2d) %s\n" "$count" "$model"
      ((count++))
    done
    echo "----------------------------------------"
    
    # Get user selection
    echo -n "Enter model number to select (or 'b' to go back): "
    read -r choice
    
    if [[ "$choice" == "b" || "$choice" == "B" ]]; then
      return 1
    fi
    
    # Validate selection
    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#model_names[@]}" ]; then
      selected_model="${model_names[$((choice-1))]}"
      echo "Selected model: $selected_model"
      return 0
    else
      echo "❌ Invalid selection"
      return 1
    fi
  else
    echo "❌ Failed to retrieve model list"
    # Show recent logs if available
    if [ -f "$LOG_FILE" ]; then
      echo ""
      echo "📝 Recent Ollama server logs:"
      echo "----------------------------------------"
      tail -n 10 "$LOG_FILE" | sed 's/^/  /'
      echo "----------------------------------------"
    fi
    return 1
  fi
}

# Function to show interactive menu
show_interactive_menu() {
  while true; do
    clear
    echo "╔══════════════════ Ollama Server Manager ═══════════════════╗"
    echo "║                                                           ║"
    echo "║  1) 🚀 Start Ollama Server                                ║"
    echo "║  2) 🛑 Stop Ollama Server                                 ║"
    echo "║  3) 📥 Download a Model                                   ║"
    echo "║  4) 📋 List Available Models                              ║"
    echo "║  5) 🧪 Test a Model                                       ║"
    echo "║  6) 🔍 Show Detailed Status                               ║"
    echo "║  7) 📝 View Log File                                      ║"
    echo "║  8) 📥 Load Model into Memory                             ║"
    echo "║  9) 📤 Unload Model from Memory                           ║"
    echo "║ 10) ℹ️  Show Help                                         ║"
    echo "║  0) ❌ Exit                                               ║"
    echo "║                                                           ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    
    # Show detailed status
    echo ""
    echo "🔍 Server Status:"
    echo "----------------------------------------"
    
    # Check installation
    if command -v ollama &>/dev/null; then
      local version=$(ollama --version 2>&1 | head -n 1)
      echo "📦 Installation: ✅ $version"
    else
      echo "📦 Installation: ❌ Not installed"
    fi
    
    # Check process status
    if [ -f "$PID_FILE" ]; then
      local pid=$(cat "$PID_FILE")
      if kill -0 "$pid" 2>/dev/null; then
        echo "🔄 Process: ✅ Running (PID: $pid)"
        # Get process details
        local process_info=$(ps -p "$pid" -o %cpu,%mem,etime,command 2>/dev/null)
        if [ -n "$process_info" ]; then
          echo "   Details:"
          echo "$process_info" | tail -n 1 | sed 's/^/   /'
        fi
      else
        echo "🔄 Process: ⚠️ Stale PID file (PID: $pid)"
        rm -f "$PID_FILE"
      fi
    else
      echo "🔄 Process: ❌ No PID file"
    fi
    
    # Check API status
    if curl -s --max-time 2 "${OLLAMA_API_URL}/tags" &>/dev/null; then
      echo "🌐 API: ✅ Responding"
      # Get server version
      local server_version=$(curl -s --max-time 2 "${OLLAMA_API_URL}/version" | jq -r '.version' 2>/dev/null)
      if [ -n "$server_version" ]; then
        echo "   Version: $server_version"
      fi
    else
      echo "🌐 API: ❌ Not responding"
    fi
    
    # Show model count and loaded models
    local models_json=$(curl -s --max-time 2 "${OLLAMA_API_URL}/tags")
    if [ -n "$models_json" ]; then
      local model_count=$(echo "$models_json" | jq -r '.models | length' 2>/dev/null || echo "0")
      echo "🤖 Models: ✅ $model_count available"
      
      # Show currently loaded models
      echo "📥 Currently Loaded Models:"
      check_loaded_models
    else
      echo "🤖 Models: ❓ Unknown"
    fi
    
    # Show system resources
    echo ""
    echo "💻 System Resources:"
    echo "  CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")"
    echo "  Memory: $(get_memory_info)"
    echo "  Disk Space: $(df -h / | awk 'NR==2 {print $4}')"
    
    echo ""
    echo "----------------------------------------"
    
    echo -n "Enter your choice [0-10]: "
    read -r choice
    
    case $choice in
      1)
        echo "Starting Ollama server..."
        start_ollama
        echo -n "Press Enter to continue..."
        read -r
        ;;
      2)
        echo "Stopping Ollama server..."
        stop_ollama
        echo -n "Press Enter to continue..."
        read -r
        ;;
      3)
        echo "Select a model to download:"
        if show_model_selection_menu; then
          download_model "$selected_model"
        fi
        echo -n "Press Enter to continue..."
        read -r
        ;;
      4)
        list_models
        echo -n "Press Enter to continue..."
        read -r
        ;;
      5)
        echo "Select a model to test:"
        if show_model_selection_menu; then
          test_model "$selected_model"
        fi
        echo -n "Press Enter to continue..."
        read -r
        ;;
      6)
        show_status
        echo -n "Press Enter to continue..."
        read -r
        ;;
      7)
        if [ -f "$LOG_FILE" ]; then
          echo "Showing last 20 lines of log file:"
          echo "----------------------------------------"
          tail -n 20 "$LOG_FILE"
        else
          echo "Log file not found: $LOG_FILE"
        fi
        echo -n "Press Enter to continue..."
        read -r
        ;;
      8)
        echo "Select a model to load into memory:"
        if show_model_selection_menu; then
          load_model "$selected_model"
        fi
        echo -n "Press Enter to continue..."
        read -r
        ;;
      9)
        echo "Select a model to unload from memory:"
        if show_model_selection_menu; then
          unload_model "$selected_model"
        fi
        echo -n "Press Enter to continue..."
        read -r
        ;;
      10)
        show_help
        echo -n "Press Enter to continue..."
        read -r
        ;;
      0)
        echo "Exiting..."
        exit 0
        ;;
      *)
        echo "Invalid choice. Press Enter to try again..."
        read -r
        ;;
    esac
  done
}

# Main script
if [ "$#" -eq 0 ]; then
  show_interactive_menu
else
  case "$1" in
    "start")
      check_ollama_installed && start_ollama
      ;;
    "stop")
      stop_ollama
      ;;
    "status")
      show_status
      ;;
    "list")
      check_ollama_running && list_models
      ;;
    "pull")
      if [ -z "$2" ]; then
        echo "Select a model to download:"
        if show_model_selection_menu; then
          download_model "$selected_model"
        fi
      else
        check_ollama_running && download_model "$2"
      fi
      ;;
    "test")
      if [ -z "$2" ]; then
        echo "Select a model to test:"
        if show_model_selection_menu; then
          test_model "$selected_model"
        fi
      else
        check_ollama_running && test_model "$2"
      fi
      ;;
    "load")
      if [ -z "$2" ]; then
        echo "Select a model to load into memory:"
        if show_model_selection_menu; then
          load_model "$selected_model"
        fi
      else
        check_ollama_running && load_model "$2"
      fi
      ;;
    "unload")
      if [ -z "$2" ]; then
        echo "Select a model to unload from memory:"
        if show_model_selection_menu; then
          unload_model "$selected_model"
        fi
      else
        check_ollama_running && unload_model "$2"
      fi
      ;;
    "help"|"--help"|"-h")
      show_help
      ;;
    *)
      echo "❌ Unknown command: $1"
      show_help
      exit 1
      ;;
  esac
fi 