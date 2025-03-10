# write me a shell script that finds all the markdown files from the current folder downwards, and concatentes them all into a single giant markdown file separated by a new line. be sure this is robust and lets the user know what it is doing as it is going. 
#!/bin/bash

output_file="all_markdown_files.md"

# Check if the output file already exists and remove it
if [ -f "$output_file" ]; then
    echo "Removing existing output file: $output_file"
    rm "$output_file"
fi

# Find all markdown files and concatenate them
find . -type f -name "*.md" | while read -r file; do
    echo "Processing $file"
    cat "$file" >> "$output_file"
    echo -e "\n" >> "$output_file"
done

echo "All markdown files have been concatenated into $output_file"