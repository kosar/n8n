#!/usr/bin/env python3

import os
import sys
import asyncio
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext, messagebox
import threading

# Import functionality from main module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from date_extractor import (
    process_directory, DEFAULT_PROMPT, DEFAULT_MODEL,
    check_ollama_model, check_requirements as check_cli_requirements
)

class RedirectText:
    """Redirect print statements to tkinter text widget"""
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.buffer = ""

    def write(self, string):
        self.buffer += string
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
        self.text_widget.update()

    def flush(self):
        pass

class DateExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Date Extractor")
        self.root.geometry("800x600")
        self.root.minsize(640, 480)
        
        # Set up variables
        self.directory_var = tk.StringVar(value=os.path.expanduser("~"))
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.prompt_var = tk.StringVar(value=DEFAULT_PROMPT)
        self.recursive_var = tk.BooleanVar(value=False)
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create and place widgets
        self.create_widgets(main_frame)
        
        # Initialize available models list
        self.available_models = []
        self.update_model_list_async()
        
    def create_widgets(self, parent):
        # Directory selection
        dir_frame = ttk.LabelFrame(parent, text="Directory", padding="5")
        dir_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(dir_frame, textvariable=self.directory_var, width=50).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(dir_frame, text="Browse", command=self.browse_directory).pack(side=tk.RIGHT)
        
        # Model selection
        model_frame = ttk.LabelFrame(parent, text="Vision Model", padding="5")
        model_frame.pack(fill=tk.X, pady=5)
        
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, width=40)
        self.model_combo.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(model_frame, text="Refresh", command=self.update_model_list_async).pack(side=tk.RIGHT)
        
        # Prompt input
        prompt_frame = ttk.LabelFrame(parent, text="Extraction Prompt", padding="5")
        prompt_frame.pack(fill=tk.X, pady=5)
        
        prompt_entry = ttk.Entry(prompt_frame, textvariable=self.prompt_var, width=50)
        prompt_entry.pack(fill=tk.X, padx=5, pady=5)
        
        # Options
        options_frame = ttk.LabelFrame(parent, text="Options", padding="5")
        options_frame.pack(fill=tk.X, pady=5)
        
        ttk.Checkbutton(options_frame, text="Process subdirectories recursively", variable=self.recursive_var).pack(anchor=tk.W)
        
        # Action buttons
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Start Processing", command=self.start_processing).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Reset", command=self.reset_form).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Exit", command=self.root.quit).pack(side=tk.RIGHT, padx=5)
        
        # Status and progress
        self.progress = ttk.Progressbar(parent, orient=tk.HORIZONTAL, length=100, mode='indeterminate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Log output
        log_frame = ttk.LabelFrame(parent, text="Log Output", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Footer
        footer = ttk.Label(parent, text="Powered by Ollama Vision Models", font=("", 8))
        footer.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
    
    def browse_directory(self):
        """Open directory browser dialog"""
        directory = filedialog.askdirectory(
            initialdir=self.directory_var.get(),
            title="Select Directory Containing Images"
        )
        if directory:
            self.directory_var.set(directory)
            
    def update_model_list_async(self):
        """Update the model list asynchronously"""
        self.progress.start()
        threading.Thread(target=self.update_model_list, daemon=True).start()
        
    def update_model_list(self):
        """Get available vision models from Ollama"""
        try:
            # Import here to avoid circular imports
            import ollama
            
            # Get models from Ollama
            models_response = ollama.list()
            
            # Filter for vision models (best guess)
            vision_models = []
            potential_vision_keywords = ['vision', 'visual', 'llava', 'llama-vision', 'bakllava', 'image']
            
            if 'models' in models_response:
                for model in models_response['models']:
                    model_name = model.get('name', '')
                    if not model_name and 'model' in model:
                        model_name = model['model']
                        
                    if any(keyword in model_name.lower() for keyword in potential_vision_keywords):
                        vision_models.append(model_name)
            
            # Also add known vision models
            known_models = [DEFAULT_MODEL, 'llava', 'llama3-vision', 'bakllava']
            for model in known_models:
                if model not in vision_models:
                    vision_models.append(model)
            
            # Update combobox values
            self.model_combo['values'] = vision_models
            self.available_models = vision_models
            
        except Exception as e:
            print(f"Error getting models: {str(e)}")
        
        finally:
            self.root.after(0, self.progress.stop)
    
    def reset_form(self):
        """Reset form to default values"""
        self.directory_var.set(os.path.expanduser("~"))
        self.model_var.set(DEFAULT_MODEL)
        self.prompt_var.set(DEFAULT_PROMPT)
        self.recursive_var.set(False)
        self.log_text.delete(1.0, tk.END)
    
    def check_requirements(self):
        """Check if all requirements are met"""
        # Check directory
        if not os.path.isdir(self.directory_var.get()):
            messagebox.showerror("Error", f"Directory '{self.directory_var.get()}' does not exist")
            return False
        
        # Check model
        if not self.model_var.get():
            messagebox.showerror("Error", "Please select a model")
            return False
            
        # Check if Ollama is running
        try:
            import ollama
            ollama.list()
        except Exception as e:
            messagebox.showerror("Error", f"Cannot connect to Ollama: {str(e)}")
            return False
        
        return True
    
    def start_processing(self):
        """Start the processing in a separate thread"""
        if not self.check_requirements():
            return
            
        # Disable buttons
        for widget in self.root.winfo_children():
            if isinstance(widget, ttk.Button):
                widget.configure(state=tk.DISABLED)
                
        # Clear log
        self.log_text.delete(1.0, tk.END)
        
        # Redirect stdout to our log widget
        old_stdout = sys.stdout
        sys.stdout = RedirectText(self.log_text)
        
        # Start progress bar
        self.progress.start()
        
        # Start processing in separate thread
        threading.Thread(
            target=self.run_processing,
            args=(
                self.directory_var.get(),
                self.prompt_var.get(),
                self.model_var.get(),
                self.recursive_var.get()
            ),
            daemon=True
        ).start()
        
        # Restore stdout
        sys.stdout = old_stdout
    
    def run_processing(self, directory, prompt, model, recursive):
        try:
            # Run the async function in the thread
            asyncio.run(process_directory(directory, prompt, model, recursive))
            
            # Show completion message
            self.root.after(0, lambda: messagebox.showinfo(
                "Processing Complete",
                f"Date extraction complete for {directory}"
            ))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Error",
                f"An error occurred: {str(e)}"
            ))
            
        finally:
            # Stop progress bar
            self.root.after(0, self.progress.stop)
            
            # Re-enable buttons
            def enable_buttons():
                for widget in self.root.winfo_children():
                    if isinstance(widget, ttk.Button):
                        widget.configure(state=tk.NORMAL)
            
            self.root.after(0, enable_buttons)

def main():
    """Start the GUI application"""
    root = tk.Tk()
    app = DateExtractorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
