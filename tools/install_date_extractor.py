#!/usr/bin/env python3

import os
import sys
import subprocess
import platform
import webbrowser
from pathlib import Path

def print_colored(text, color_code):
    """Print colored text to terminal"""
    print(f"\033[{color_code}m{text}\033[0m")

def print_info(message):
    print_colored(f"INFO: {message}", "36")

def print_success(message):
    print_colored(f"✓ {message}", "32")

def print_error(message):
    print_colored(f"ERROR: {message}", "31")

def print_warning(message):
    print_colored(f"WARNING: {message}", "33")

def install_python_packages():
    """Install required Python packages"""
    print_info("Installing required Python packages...")
    
    requirements_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requirements.txt")
    
    if not os.path.exists(requirements_file):
        print_error(f"Requirements file not found: {requirements_file}")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
        print_success("Python packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to install packages: {str(e)}")
        return False

def check_python_version():
    """Check if Python version is adequate"""
    min_version = (3, 6)
    current_version = sys.version_info
    
    if current_version < min_version:
        print_error(f"Python {min_version[0]}.{min_version[1]} or higher is required")
        print_info(f"Current version: Python {current_version[0]}.{current_version[1]}.{current_version[2]}")
        return False
    
    print_success(f"Python version {current_version[0]}.{current_version[1]}.{current_version[2]} detected")
    return True

def check_ollama_installation():
    """Check if Ollama is installed and running"""
    try:
        # Try importing ollama
        import ollama
        
        # Try connecting to Ollama server
        try:
            ollama.list()
            print_success("Ollama is installed and running")
            return True
        except Exception as e:
            print_warning(f"Ollama package is installed, but server connection failed: {str(e)}")
            print_info("Make sure Ollama server is running")
            return False
            
    except ImportError:
        print_warning("Ollama Python package not found")
        return False

def offer_ollama_installation():
    """Offer to open Ollama installation page"""
    print_info("Ollama is required for this tool to work")
    
    system = platform.system().lower()
    
    install_urls = {
        "darwin": "https://ollama.com/download/mac",
        "windows": "https://ollama.com/download/windows",
        "linux": "https://ollama.com/download/linux",
    }
    
    if system in install_urls:
        print_info(f"Installation instructions for {system.capitalize()}: {install_urls[system]}")
        response = input("Open installation page in browser? (y/n): ").lower()
        if response == 'y' or response == 'yes':
            webbrowser.open(install_urls[system])
    else:
        print_info("Visit https://ollama.com/download for installation instructions")

def check_vision_models():
    """Check for available vision models"""
    print_info("Checking for vision models...")
    
    try:
        import ollama
        models_response = ollama.list()
        
        if 'models' not in models_response:
            print_warning("Unexpected response format from Ollama API")
            return False
        
        vision_models = []
        potential_vision_keywords = ['vision', 'visual', 'llava', 'llama-vision', 'bakllava', 'image']
        
        for model in models_response['models']:
            model_name = model.get('name', '')
            if not model_name and 'model' in model:
                model_name = model['model']
                
            if any(keyword in model_name.lower() for keyword in potential_vision_keywords):
                vision_models.append(model_name)
        
        if vision_models:
            print_success(f"Found vision models: {', '.join(vision_models)}")
            return True
        else:
            print_warning("No vision models detected")
            print_info("To download a vision model, run: ollama pull llama3.2-vision")
            
            response = input("Download llama3.2-vision model now? (y/n): ").lower()
            if response == 'y' or response == 'yes':
                try:
                    print_info("Downloading llama3.2-vision model (this may take a while)...")
                    subprocess.check_call(["ollama", "pull", "llama3.2-vision"])
                    print_success("Model downloaded successfully")
                    return True
                except subprocess.CalledProcessError as e:
                    print_error(f"Failed to download model: {str(e)}")
                    return False
                except FileNotFoundError:
                    print_error("Ollama command not found in PATH")
                    return False
            
            return False
            
    except ImportError:
        print_error("Ollama Python package not installed")
        return False
    except Exception as e:
        print_error(f"Error checking models: {str(e)}")
        return False

def create_launch_script():
    """Create a launch script for easy execution"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if platform.system().lower() == "windows":
        # Create .bat file for Windows
        bat_path = os.path.join(script_dir, "run_date_extractor.bat")
        with open(bat_path, "w") as f:
            f.write("@echo off\n")
            f.write("echo Starting Date Extractor...\n")
            f.write(f'"{sys.executable}" "{os.path.join(script_dir, "date_extractor.py")}"\n')
            f.write("pause\n")
        
        print_success(f"Created Windows batch file: {bat_path}")
        
    else:
        # Create shell script for Unix-like systems
        sh_path = os.path.join(script_dir, "run_date_extractor.sh")
        with open(sh_path, "w") as f:
            f.write("#!/bin/bash\n")
            f.write('echo "Starting Date Extractor..."\n')
            f.write(f'"{sys.executable}" "{os.path.join(script_dir, "date_extractor.py")}"\n')
        
        # Make script executable
        os.chmod(sh_path, 0o755)
        
        print_success(f"Created shell script: {sh_path}")

def main():
    """Main installation flow"""
    print_colored("=== Date Extractor Installation ===", "1;36")
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install Python packages
    install_python_packages()
    
    # Check Ollama
    if not check_ollama_installation():
        offer_ollama_installation()
    
    # Check vision models
    check_vision_models()
    
    # Create launch script
    create_launch_script()
    
    print_colored("\n=== Installation Complete ===", "1;32")
    print_info("To run the application:")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if platform.system().lower() == "windows":
        print_info(f"  - Double-click {os.path.join(script_dir, 'run_date_extractor.bat')}")
    else:
        print_info(f"  - Run: {os.path.join(script_dir, 'run_date_extractor.sh')}")
    
    print_info("  - Or from command line: python date_extractor.py")
    print_info("  - For command line options: python cli_date_extractor.py --help")
    print_info("  - For GUI interface: python gui_date_extractor.py")
    
    print_colored("\nEnjoy using Date Extractor!", "1;36")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_colored("\nInstallation cancelled by user", "33")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        sys.exit(1)
