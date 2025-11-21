import ollama
import json

# --- 1. Define Your Inputs ---
input_sentence = "woman holding bottle"
image_description = """The image features a man and a woman standing side by side. They are holding water bottles, with the woman holding a blue water bottle and the man holding a red water bottle. Both bottles are filled with water. The woman is wearing a red shirt, while the man is wearing a blue shirt. They are standing in front of a gray background."""

# --- 2. Prompts for the FIRST Call (Query Parser) ---
# This new prompt implements your recommendation.
# It parses the sentence into a subject and object.
parser_system_prompt = """You are an AI language parser. Your task is to read a `FOCUS_QUERY` and extract the main 'subject' and the 'object_of_action' into a JSON format.

**TASK:**
1.  Read the `FOCUS_QUERY` (e.g., "woman holding bottle").
2.  Identify the main subject (who is doing the action).
3.  Identify the main object (what is being acted upon).
4.  Output *only* a JSON object in the format: {"subject": "...", "object_of_action": "..."}

**CONSTRAINTS:**
* Your output MUST be *only* the JSON object.
* Do not write "json", use code blocks, or add any conversational text.
* If there is no clear object, set "object_of_action" to "null".

**EXAMPLE 1:**
* `FOCUS_QUERY`: "man and his dog"
* `EXPECTED OUTPUT`: {"subject": "man", "object_of_action": "dog"}

**EXAMPLE 2:**
* `FOCUS_QUERY`: "woman holding bottle"
* `EXPECTED OUTPUT`: {"subject": "woman", "object_of_action": "bottle"}
"""

parser_user_prompt = f"FOCUS_QUERY: \"{input_sentence}\""

# --- 3. Prompts for the SECOND Call (Sentence Generator) ---
# This prompt now receives the structured JSON from Step 1.
segmenter_system_prompt = """You are an AI data extractor. Your task is to perform a specific analysis and return *only* a JSON object.

**TASK:**
1.  Read the `IMAGE_DESCRIPTION` and the `OBJECTS_TO_DESCRIBE` JSON.
2.  The `OBJECTS_TO_DESCRIBE` contains a "subject" and an "object_of_action".
3.  You must generate one comprehensive sentence for the "subject".
4.  You must generate one comprehensive sentence for the "object_of_action".
5.  For *each* sentence, find all information about that object from the `IMAGE_DESCRIPTION`.
6.  **Crucially, use the relationship** (e.g., "woman" and "bottle") as context to find the *correct* object in the image.
7.  Each sentence **must** start with the exact prefix: `Segment this object which...`
8.  Format this output *only* as a JSON object: {sentence1: "...", sentence2: "..."}

**CONSTRAINTS:**
* Your output MUST be *only* the JSON object.
* Do not write "json", use code blocks, or add any conversational text.
* All information must come *only* from the `IMAGE_DESCRIPTION`.
"""

# --- 4. Main Execution ---
try:
    # --- CALL 1: Parse the input_sentence into structured JSON ---
    print(f"--- Step 1: Parsing query '{input_sentence}' ---")
    
    parser_response = ollama.chat(
        model='llama3:8b',
        messages=[
            {'role': 'system', 'content': parser_system_prompt},
            {'role': 'user', 'content': parser_user_prompt},
        ]
    )
    
    # Get the raw JSON string from the model
    object_json_string = parser_response['message']['content'].strip()
    print(f"--- Objects identified: {object_json_string} ---")
    
    # Safely convert the string JSON into a Python dict
    try:
        object_dict = json.loads(object_json_string)
        if not isinstance(object_dict, dict):
            raise ValueError("Model did not return a dictionary.")
            
    except Exception as e:
        print(f"\n--- ERROR: Could not parse the object JSON from the model. ---")
        print(f"Model returned: {object_json_string}")
        print(f"Error: {e}")
        exit()

    # --- CALL 2: Generate segmentation sentences using the new dict ---
    print(f"--- Step 2: Generating segmentation sentences for {object_dict} ---")
    
    # Build the final user prompt for the segmenter
    segmenter_user_prompt = f"""IMAGE_DESCRIPTION: "{image_description}"
<|im_end|>
OBJECTS_TO_DESCRIBE: {object_json_string}
"""

    # Run the second call
    final_response = ollama.chat(
        model='llama3:8b',
        messages=[
            {'role': 'system', 'content': segmenter_system_prompt},
            {'role': 'user', 'content': segmenter_user_prompt},
        ]
    )
    
    # --- FINAL OUTPUT ---
    print("\n--- FINAL JSON OUTPUT ---")
    print(final_response['message']['content'])


except Exception as e:
    print("\n--- PYTHON SCRIPT ERROR ---")
    print("The code failed. Please check the following:")
    print("1. Is the 'ollama' library installed? (Run: pip install ollama)")
    print("2. Is the Ollama application running on your computer?")
    print("3. Have you pulled the 'llama3:8b' model? (Run: ollama pull llama3:8b)")
    print("\nError details:")
    print(e)