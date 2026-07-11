import os
import re
from bs4 import BeautifulSoup

# Define paths for Windows compatibility
RAW_DIR = os.path.join("data", "raw")
CLEAN_DIR = os.path.join("data", "clean")

# Phrases to drop completely to remove website UI noise
BANNED_PHRASES = [
    "something went wrong", "please try again later", "sign in to apply",
    "are you sure you want to sign out", "you need to sign in before applying",
    "redirected to an external website", "jan samarth", "accessibility options",
    "dyslexia friendly", "adhd mode", "text to speech", "invert colors",
    "bigger text", "smaller text", "text spacing", "line height", "saturation",
    "low saturation", "high saturation", "desaturate", "light-dark", "highlight links",
    "pause animation", "hide images", "reset all settings", "powered by",
    "digital india corporation", "connect on social media", "quick links",
    "screen reader", "accessibility statement"
]

def should_filter_line(line):
    """Returns True if the line is just website UI noise."""
    lowered = line.lower().strip()
    if not lowered:
        return True
    # Check against our denylist of UI phrases
    for phrase in BANNED_PHRASES:
        if phrase in lowered:
            return True
    return False

def clean_html_file(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Remove non-content structural elements entirely
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'button']):
        element.decompose()

    # Get raw text breakdown
    raw_text = soup.get_text(separator='\n')
    
    # Clean text line-by-line
    cleaned_lines = []
    for line in raw_text.splitlines():
        cleaned_line = line.strip()
        if cleaned_line and not should_filter_line(cleaned_line):
            cleaned_lines.append(cleaned_line)

    # Reconstruct text block and compress erratic white spaces
    final_text = '\n'.join(cleaned_lines)
    final_text = re.sub(re.compile(r'\n{3,}'), '\n\n', final_text)

    # Ensure output directory exists and write file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_text)
    
    print(f"Cleaned: {os.path.basename(file_path)} -> Saved to {output_path}")

def process_all_html():
    if not os.path.exists(RAW_DIR):
        print(f"Error: Raw directory '{RAW_DIR}' not found. Please ensure your folders are set up.")
        return

    file_count = 0
    for root, dirs, files in os.walk(RAW_DIR):
        # Ignore browser-generated asset folders (e.g., scheme_files)
        if "_files" in root or ".txt" in root:
            continue
            
        for file in files:
            if file.endswith(".html") or file.endswith(".htm"):
                raw_file_path = os.path.join(root, file)
                
                # Maintain the same folder hierarchy under data/clean/
                relative_path = os.path.relpath(root, RAW_DIR)
                clean_file_name = os.path.splitext(file)[0] + ".txt"
                clean_file_path = os.path.join(CLEAN_DIR, relative_path, clean_file_name)
                
                clean_html_file(raw_file_path, clean_file_path)
                file_count += 1
                
    print(f"\nProcessing complete. Successfully cleaned {file_count} files.")

if __name__ == "__main__":
    print("Starting data extraction and text cleaning...")
    process_all_html()