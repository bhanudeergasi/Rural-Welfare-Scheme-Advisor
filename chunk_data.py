import os
import json

CLEAN_DIR = os.path.join("data", "clean")
OUTPUT_CHUNKS_FILE = "data_chunks.json"

# The exact section headers we expect to see in the myScheme extractions
SECTION_HEADERS = [
    "Details", "Benefits", "Eligibility", "Exclusions", 
    "Application Process", "Documents Required", "Frequently Asked Questions"
]

def get_scheme_id_from_filename(filename):
    """
    Cleans up the filename to match our standard scheme IDs.
    You might need to tweak this if your text file names differ significantly.
    """
    base_name = os.path.splitext(filename)[0].lower()
    return base_name.replace("-", "_")

def chunk_file(file_path):
    filename = os.path.basename(file_path)
    scheme_id = get_scheme_id_from_filename(filename)
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    chunks = []
    current_section = "details" # Default starting state
    current_content = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        # If the line matches a known header exactly, process the previous chunk
        if any(stripped.lower() == header.lower() for header in SECTION_HEADERS):
            if current_content:
                chunks.append({
                    "scheme_name": scheme_id,
                    "section_type": current_section.lower().replace(" ", "_"),
                    "content": "\n".join(current_content).strip(),
                    "source_url": f"https://www.myscheme.gov.in/schemes/{scheme_id.replace('_', '-')}",
                    "last_verified_date": "2026-07-09",
                    "state_specific": None
                })
            # Reset state for the new section
            current_section = stripped
            current_content = []
        else:
            current_content.append(stripped)

    # Flush out the final section trailing at the end of the file
    if current_content:
        chunks.append({
            "scheme_name": scheme_id,
            "section_type": current_section.lower().replace(" ", "_"),
            "content": "\n".join(current_content).strip(),
            "source_url": f"https://www.myscheme.gov.in/schemes/{scheme_id.replace('_', '-')}",
            "last_verified_date": "2026-07-09",
            "state_specific": None
        })

    return chunks

def process_all():
    all_chunks = []
    if not os.path.exists(CLEAN_DIR):
        print(f"Error: Directory '{CLEAN_DIR}' not found. Check your folder structure.")
        return

    for root, dirs, files in os.walk(CLEAN_DIR):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                file_chunks = chunk_file(file_path)
                all_chunks.extend(file_chunks)
                print(f"Processed {file}: Generated {len(file_chunks)} structural chunks.")

    # Save the master JSON array
    with open(OUTPUT_CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=4, ensure_ascii=False)
        
    print(f"\nSuccess! Compiled {len(all_chunks)} chunks into '{OUTPUT_CHUNKS_FILE}'.")

if __name__ == "__main__":
    print("Starting structure-aware chunking...")
    process_all()