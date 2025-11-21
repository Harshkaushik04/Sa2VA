import os
import sys

def shift_folders_down(root_path):
    # Verify the path exists
    if not os.path.exists(root_path):
        print(f"Error: The path '{root_path}' does not exist.")
        return

    print(f"Processing folders in: {root_path}")
    print("-" * 30)

    # We iterate from 3 up to 51 (range is exclusive at the end, so we use 52)
    # It is crucial to do this in ascending order (3, then 4, then 5)
    # to ensure we don't overwrite a folder we just renamed.
    for i in range(3, 52):
        old_name = str(i)
        new_name = str(i - 1)

        old_path = os.path.join(root_path, old_name)
        new_path = os.path.join(root_path, new_name)

        # Check if the source folder (e.g., '3') exists
        if os.path.exists(old_path):
            # Safety Check: Ensure the destination (e.g., '2') is empty/free
            if os.path.exists(new_path):
                print(f"[SKIP] Cannot rename '{old_name}' to '{new_name}'. Folder '{new_name}' already exists.")
            else:
                try:
                    os.rename(old_path, new_path)
                    print(f"[OK] Renamed: '{old_name}' -> '{new_name}'")
                except Exception as e:
                    print(f"[ERROR] Could not rename '{old_name}': {e}")
        else:
            print(f"[MISSING] Folder '{old_name}' not found, skipping.")

    print("-" * 30)
    print("Renaming complete.")

if __name__ == "__main__":
    # Ask user for input
    target_folder = input("Enter the full path to the parent folder: ").strip()
    
    # Remove quotes if the user copied path as "C:\Path"
    target_folder = target_folder.replace('"', '').replace("'", "")
    
    confirm = input("This will rename folders 3...51 to 2...50. Proceed? (y/n): ")
    if confirm.lower() == 'y':
        shift_folders_down(target_folder)
    else:
        print("Operation cancelled.")