import json

def merge_files(file1_path, file2_path, output_path):
    try:
        # 1. Read the input files
        print("Reading files...")
        with open(file1_path, 'r', encoding='utf-8') as f1:
            data1 = json.load(f1)
            
        with open(file2_path, 'r', encoding='utf-8') as f2:
            data2 = json.load(f2)

        # Dictionary to hold the final result
        merged_data = {}

        # 2. Find keys present in BOTH files (Intersection)
        # We convert keys to sets to easily find the intersection
        keys1 = set(data1.keys())
        keys2 = set(data2.keys())
        
        common_keys = keys1.intersection(keys2)
        
        print(f"Found {len(common_keys)} common keys to process.")

        # 3. Process only the common keys
        for key in common_keys:
            # We use data2 as the 'base' because the prompt says 
            # "keep everything else same" and File 2 has more attributes/keys.
            merged_object = data2[key]
            
            # Get FOCUS_QUERY from File 1
            if 'FOCUS_QUERY' in data1[key]:
                merged_object['FOCUS_QUERY'] = data1[key]['FOCUS_QUERY']
            
            # Ensure IMAGE_DESCRIPTION is from File 2 
            # (This is already true since we are using data2 as base, 
            # but we explicitly check to ensure it exists).
            if 'IMAGE_DESCRIPTION' in data2[key]:
                merged_object['IMAGE_DESCRIPTION'] = data2[key]['IMAGE_DESCRIPTION']
            
            # Add to final result
            merged_data[key] = merged_object

        # 4. Write the output to the third file
        with open(output_path, 'w', encoding='utf-8') as out:
            json.dump(merged_data, out, indent=4)
            
        print(f"Success! Merged data written to '{output_path}'")

    except json.JSONDecodeError:
        print("Error: One of the files is not valid JSON. Please check the file format.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- CONFIGURATION ---
# Replace these with your actual file names
input_file_1 = '/home/harsh/AI/Sa2VA/Sa2VA/my_results.txt'  # File with FOCUS_QUERY (fewer keys)
input_file_2 = '/home/harsh/AI/Sa2VA/Sa2VA/my_gemini_results.txt'  # File with IMAGE_DESCRIPTION (more keys)
output_file = '/home/harsh/AI/Sa2VA/Sa2VA/new_gemini_output.txt'

# Run the function
if __name__ == "__main__":
    merge_files(input_file_1, input_file_2, output_file)