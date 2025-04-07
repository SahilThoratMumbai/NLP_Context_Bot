import streamlit as st
import nltk
import os
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

# ========== Setup ==========
# Configure NLTK data path
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK data
required_data = [
    ('punkt', 'tokenizers/punkt'),
    ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
    ('wordnet', 'corpora/wordnet'),
    ('omw-1.4', 'corpora/omw-1.4')
]

for resource, path in required_data:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_path)

# Custom spelling correction to avoid SpellChecker dependency
class SimpleSpellChecker:
    def __init__(self):
        self.common_words = {
            'hello', 'hi', 'bank', 'book', 'love', 'river',
            'financial', 'institution', 'knowledge', 'emotion'
        }
    
    def correction(self, word):
        return word  # Simple implementation - doesn't actually correct
    
    def __contains__(self, word):
        return word.lower() in self.common_words

spell = SimpleSpellChecker()

# ========== NLP Helpers ==========
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

def correct_spelling(text):
    tokens = word_tokenize(text)
    return ' '.join(tokens)  # Skip actual correction for now

def simple_lesk_definition(word, context_sentence, pos=None):
    max_overlap = 0
    best_sense = None
    context = set(word_tokenize(context_sentence.lower()))
    
    for sense in wn.synsets(word, pos=pos):
        signature = set()
        signature.update(word_tokenize(sense.definition().lower()))
        for example in sense.examples():
            signature.update(word_tokenize(example.lower()))
        
        overlap = len(context.intersection(signature))
        if overlap > max_overlap:
            max_overlap = overlap
            best_sense = sense
    
    return best_sense.definition() if best_sense else None

def process_input(user_input):
    corrected = correct_spelling(user_input)
    tokens = word_tokenize(corrected)
    tags = pos_tag(tokens)
    senses = {}

    for word, tag in tags:
        wn_pos = get_wordnet_pos(tag)
        if word.lower() == "bank":
            senses[word] = "financial institution" if "money" in corrected.lower() else "river bank"
        else:
            definition = simple_lesk_definition(word, corrected, wn_pos)
            if definition:
                senses[word] = definition
    
    return corrected, tags, senses

def generate_response(corrected, pos_tags, senses):
    lowered = corrected.lower()
    if "bank" in senses:
        return "Talking about finances?" if "financial" in senses["bank"] else "Nice river view!"
    elif any(w in lowered for w in ["book", "read"]):
        return "I love reading too!"
    elif "love" in lowered:
        return "Love is wonderful!"
    return "Interesting! Tell me more."

# ========== Streamlit UI ==========
st.set_page_config(page_title="NLP ContextBot", page_icon="🧠")
st.title("🧠 NLP ContextBot")
st.markdown("This bot demonstrates word sense disambiguation using WordNet.")

user_input = st.text_input("You:", key="input")

if user_input:
    try:
        corrected, tags, senses = process_input(user_input)
        
        st.markdown("### Processing Results")
        st.json({
            "Corrected Text": corrected,
            "POS Tags": tags,
            "Word Senses": senses
        })
        
        response = generate_response(corrected, tags, senses)
        st.markdown(f"### 🤖 Bot: {response}")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("Note: The bot is using simplified text processing for this demo.")