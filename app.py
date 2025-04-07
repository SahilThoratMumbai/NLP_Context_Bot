import streamlit as st
import nltk
import os
import re
from nltk.corpus import wordnet as wn

# ========== Setup with Comprehensive Fallbacks ==========
class FallbackTokenizer:
    """Simple regex-based tokenizer as fallback"""
    @staticmethod
    def tokenize(text):
        return re.findall(r"\w+(?:'\w+)?|\S", text)

# Configure NLTK with multiple fallback options
nltk_data_paths = [
    os.path.join(os.getcwd(), "nltk_data"),  # Current directory
    "/tmp/nltk_data",                        # Temp directory
    os.path.join(os.path.expanduser("~"), "nltk_data")  # Home directory
]

for path in nltk_data_paths:
    try:
        os.makedirs(path, exist_ok=True)
        nltk.data.path.append(path)
    except:
        continue

# Download critical resources with retries
resources = [
    ('punkt', 'tokenizers/punkt'),
    ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger'),
    ('wordnet', 'corpora/wordnet'),
    ('omw-1.4', 'corpora/omw-1.4')
]

for resource, path in resources:
    max_retries = 2
    for _ in range(max_retries):
        try:
            nltk.data.find(path)
            break
        except LookupError:
            try:
                nltk.download(resource, download_dir=nltk_data_paths[0])
                break
            except:
                continue

# ========== NLP Functions with Fallbacks ==========
def get_tokenizer():
    """Returns the best available tokenizer"""
    try:
        nltk.data.find('tokenizers/punkt')
        return nltk.word_tokenize
    except:
        return FallbackTokenizer.tokenize

word_tokenize = get_tokenizer()

def get_wordnet_pos(treebank_tag):
    mappings = {
        'J': wn.ADJ,
        'V': wn.VERB,
        'N': wn.NOUN,
        'R': wn.ADV
    }
    return mappings.get(treebank_tag[0], wn.NOUN)

def process_text(text):
    tokens = word_tokenize(text)
    tags = nltk.pos_tag(tokens) if hasattr(nltk, 'pos_tag') else [(token, '') for token in tokens]
    
    senses = {}
    for word, tag in tags:
        wn_pos = get_wordnet_pos(tag)
        if word.lower() == "bank":
            senses[word] = "financial" if any(w in text.lower() for w in ["money", "account"]) else "river"
        else:
            try:
                synsets = wn.synsets(word, pos=wn_pos)
                if synsets:
                    senses[word] = synsets[0].definition()
            except:
                continue
    
    return tokens, tags, senses

# ========== Streamlit UI ==========
st.set_page_config(page_title="NLP ContextBot", page_icon="🧠")
st.title("🧠 NLP ContextBot")

user_input = st.text_input("Enter your text (type 'exit' to quit):")

if user_input and user_input.lower() != 'exit':
    try:
        with st.spinner("Analyzing..."):
            tokens, tags, senses = process_text(user_input)
            
            st.subheader("Results")
            st.json({
                "tokens": tokens,
                "tags": tags,
                "senses": senses
            })
            
            if senses:
                st.success("Analysis completed successfully!")
            else:
                st.info("No word senses detected - try more specific words")
                
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
        st.info("This simple version uses fallback methods when NLTK resources aren't available")

elif user_input.lower() == 'exit':
    st.success("Goodbye! Refresh the page to start over.")