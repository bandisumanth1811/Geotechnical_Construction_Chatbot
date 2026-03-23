"""
split_index.py
--------------
Splits the large index.faiss file (>100MB) into smaller chunks (~40MB)
so it can be safely pushed to standard GitHub without Git LFS.
"""

import os

VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "vectorstore")
FAISS_FILE = os.path.join(VECTORSTORE_DIR, "index.faiss")
CHUNK_SIZE_BYTES = 40 * 1024 * 1024  # 40 MB chunks

def split_file():
    if not os.path.exists(FAISS_FILE):
        print(f"❌ '{FAISS_FILE}' not found!")
        return

    file_size = os.path.getsize(FAISS_FILE)
    print(f"📦 Found 'index.faiss' ({file_size / (1024*1024):.2f} MB)")

    with open(FAISS_FILE, "rb") as f:
        part_num = 0
        while True:
            chunk = f.read(CHUNK_SIZE_BYTES)
            if not chunk:
                break
            
            part_name = os.path.join(VECTORSTORE_DIR, f"index_part_{part_num}.faiss")
            with open(part_name, "wb") as chunk_file:
                chunk_file.write(chunk)
            
            print(f"✅ Created {os.path.basename(part_name)} ({len(chunk) / (1024*1024):.2f} MB)")
            part_num += 1

    print(f"\n🎉 Split complete! Created {part_num} parts.")
    print("\n⚠️ IMPORTANT NEXT STEPS:")
    print("1. Delete the original 'index.faiss' file or add it to .gitignore.")
    print("2. Run 'git add vectorstore/index_part_*.faiss'")
    print("3. Uninstall Git LFS or remove the fastidous tracking.")

if __name__ == "__main__":
    split_file()
