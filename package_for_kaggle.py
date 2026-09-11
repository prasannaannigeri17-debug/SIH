import os
import zipfile

def create_kaggle_zip(source_dir, output_filename):
    print(f"Packaging project for Kaggle: {output_filename}...")
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Exclude unwanted directories
            dirs[:] = [d for d in dirs if d not in ['__pycache__', '.git', '.pytest_cache', 'venv', 'env']]
            
            for file in files:
                if file.endswith('.zip') or file.endswith('.pyc'):
                    continue
                    
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                zipf.write(file_path, arcname)
    print("Done! You can now upload this zip to Kaggle Data.")

if __name__ == "__main__":
    create_kaggle_zip(".", "SIH_Project_Kaggle.zip")
