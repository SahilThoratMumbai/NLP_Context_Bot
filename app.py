import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from pywsd.lesk import cosine_lesk  # ✅ smarter Lesk
from spellchecker import SpellChecker

# Download NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

# Initialize spell checker
spell = SpellChecker()

# POS tag converter
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
        return wn.NOUN

# Spelling correction
def correct_spelling(text):
    tokens = word_tokenize(text)
    corrected_tokens = [spell.correction(word) if word not in spell else word for word in tokens]
    return ' '.join(corrected_tokens)

# Main NLP processing
def process_input(user_input):
    corrected = correct_spelling(user_input)
    tokens = word_tokenize(corrected)
    pos_tags = pos_tag(tokens)
    disambiguated = {}

    for word, tag in pos_tags:
        wn_pos = get_wordnet_pos(tag)
        # Manual override for "bank" near "river"
        if word.lower() == "bank" and "river" in [t.lower() for t in tokens]:
            disambiguated[word] = "sloping land (especially the slope beside a body of water)"
        else:
            context = ' '.join(tokens)  # ✅ FIX: Convert token list to string
            sense = cosine_lesk(context, word, pos=wn_pos)
            if sense:
                disambiguated[word] = sense.definition()
    return corrected, pos_tags, disambiguated

# Response generation
def generate_response(corrected, pos_tags, senses):
    if "bank" in corrected:
        meaning = senses.get("bank", "")
        if "financial" in meaning or "money" in meaning:
            return "Are you talking about a financial institution?"
        elif "river" in meaning or "slope" in meaning:
            return "Oh! You mean a river bank. Sounds peaceful."
        else:
            return "Which type of bank are you referring to?"
    elif "book" in corrected:
        return "Books are a great source of knowledge!"
    elif "love" in corrected:
        return "Love is a beautiful emotion. Tell me more!"
    else:
        return "Thanks for sharing! What else would you like to talk about?"

# Streamlit UI
st.title("🧠 NLP ContextBot")
st.write("This chatbot performs spelling correction, POS tagging, and improved word sense disambiguation.")

user_input = st.text_input("You:", key="input")

if user_input:
    if user_input.lower() == 'exit':
        st.write("Bot: Goodbye!")
    else:
        corrected, pos_tags, senses = process_input(user_input)
        response = generate_response(corrected, pos_tags, senses)

        st.markdown(f"**Corrected Input:** `{corrected}`")
        st.markdown(f"**POS Tags:** `{pos_tags}`")
        st.markdown(f"**Word Senses:** `{senses}`")
        st.markdown(f"### 🤖 Bot: {response}")
