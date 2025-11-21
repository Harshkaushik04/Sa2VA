import ollama

# 1. This system prompt commands the model to find *all* distinct objects
#    and to use descriptions to separate them.
system_prompt = """You are an AI data extractor. Your task is to read an `IMAGE_DESCRIPTION` and output a list of all distinct, individual objects.

**TASK:**
1.  Read the `IMAGE_DESCRIPTION`.
2.  Identify every single object.
3.  If the description mentions multiple objects of the same type (e.g., "a man in a red shirt" and "a man in a yellow hat"), you **must** list each one as a separate item.
4.  Use the object's description from the text to make it unique in the list.

**CONSTRAINTS:**
* Your output MUST be *only* a Python list of strings.
* Do not write "python", use code blocks, or add any conversational text or explanations.
* Do not describe interactions; just list the unique object.

**EXAMPLE:**
* `IMAGE_DESCRIPTION`: "A man in a red hat is walking a brown dog. A woman in a blue coat is nearby."
* `EXPECTED OUTPUT`: ["man in a red hat", "brown dog", "woman in a blue coat"]
"""

# 2. This user prompt includes a test case with two different men and
#    three different bottles to ensure the logic works.
user_prompt = """IMAGE_DESCRIPTION: "The image features a man and a woman standing side by side. The woman is wearing a red shirt and holding a blue water bottle. The man is wearing a blue shirt and holding a red water bottle. Both of them are holding their water bottles up, showcasing them.<|im_end|>
"""

try:
    # 3. Call the Ollama chat function
    response = ollama.chat(
        model='llama3:8b',  # Using your specified model
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ]
    )
    
    # 4. Print the raw list output from the model
    print(response['message']['content'])

except Exception as e:
    print("--- PYTHON SCRIPT ERROR ---")
    print("The code failed. Please check the following:")
    print("1. Is the 'ollama' library installed? (Run: pip install ollama)")
    print("2. Is the Ollama application running on your computer?")
    print("3. Have you pulled the 'llama3:8b' model? (Run: ollama pull llama3:8b)")
    print("\nError details:")
    print(e)
