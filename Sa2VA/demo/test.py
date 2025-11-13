import spacy

# Load a small English model
nlp = spacy.load("en_core_web_sm")

text = "red bottle and green bottle"
doc = nlp(text)

# Extract only noun phrases (e.g., "red bottle", "green bottle")
nouns = [chunk.text for chunk in doc.noun_chunks]

print(", ".join(nouns))