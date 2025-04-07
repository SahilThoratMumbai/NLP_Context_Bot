import streamlit as st
import nltk
import os
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize

# ========== Setup ==========
# Configure NLTK data path - works in both local and cloud environments
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK data with explicit path handling
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', download_dir=nltk_data_path)

# Force NLTK to use our downloaded data
nltk.data.path = [nltk_data_path] + nltk.data.path

# ========== NLP Functions ==========
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

def process_text(text):
    tokens = word_tokenize(text)
    tags = pos_tag(tokens)
    senses = {}
    
    for word, tag in tags:
        wn_pos = get_wordnet_pos(tag)
        if word.lower() == "bank":
            senses[word] = "financial institution" if "money" in text.lower() else "river bank"
        else:
            synsets = wn.synsets(word, pos=wn_pos)
            if synsets:
                senses[word] = synsets[0].definition()
    
    return tokens, tags, senses

# ========== Streamlit App ==========
st.set_page_config(page_title="NLP ContextBot", page_icon="🧠")
st.title("🧠 NLP ContextBot")

user_input = st.text_input("Enter your text:")

if user_input:
    try:
        tokens, tags, senses = process_text(user_input)
        
        st.subheader("Analysis Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Tokens:**", tokens)
            st.write("**POS Tags:**", tags)
        
        with col2:
            st.write("**Word Senses:**")
            for word, sense in senses.items():
                st.write(f"- {word}: {sense}")
        
        if "bank" in senses:
            st.success("Financial context detected!" if "financial" in senses["bank"] 
                      else "Nature context detected!")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.info("If you see NLTK resource errors, try refreshing the page.")