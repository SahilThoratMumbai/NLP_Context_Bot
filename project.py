# Install required libraries


# Import libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from spellchecker import SpellChecker

# Download NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Initialize SpellChecker
spell = SpellChecker()

# Define POS tag converter for WordNet
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return wn.NOUN  # default

# Function to correct spelling
def correct_spelling(text):
    tokens = word_tokenize(text)
    corrected_tokens = [spell.correction(word) if word not in spell else word for word in tokens]
    return ' '.join(corrected_tokens)

# Function for WSD + POS
def process_input(user_input):
    # Step 1: Correct spelling
    corrected = correct_spelling(user_input)
    
    # Step 2: Tokenize and POS tag
    tokens = word_tokenize(corrected)
    pos_tags = pos_tag(tokens)
    
    # Step 3: Word Sense Disambiguation
    disambiguated = {}
    for word, tag in pos_tags:
        wn_pos = get_wordnet_pos(tag)
        sense = lesk(tokens, word, pos=wn_pos)
        if sense:
            disambiguated[word] = sense.definition()
    
    return corrected, pos_tags, disambiguated

# Response Generator
def generate_response(corrected, pos_tags, senses):
    # Rule-based dummy responses
    if "bank" in corrected:
        meaning = senses.get("bank", "")
        if "financial" in meaning or "money" in meaning:
            return "Are you talking about a financial institution?"
        elif "river" in meaning:
            return "Oh! You mean a river bank. Sounds peaceful."
        else:
            return "Which type of bank are you referring to?"
    
    elif "book" in corrected:
        return "Books are a great source of knowledge!"
    
    elif "love" in corrected:
        return "Love is a beautiful emotion. Tell me more!"
    
    else:
        return "Thanks for sharing! What else would you like to talk about?"

# Main Chat Function
def chatbot():
    print("Hello! I'm your context-aware chatbot. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Bot: Goodbye!")
            break
        corrected, pos_tags, senses = process_input(user_input)
        print(f"\n[Corrected]: {corrected}")
        print(f"[POS Tags]: {pos_tags}")
        print(f"[Word Senses]: {senses}")
        response = generate_response(corrected, pos_tags, senses)
        print(f"Bot: {response}\n")

# Run chatbot
chatbot()