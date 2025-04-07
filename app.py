import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from spellchecker import SpellChecker
import os
import pandas as pd
import time
from collections import defaultdict

# ========== Setup ==========
# Configure NLTK data path
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required resources
required_packages = [
    'punkt', 'averaged_perceptron_tagger',
    'wordnet', 'omw-1.4', 'stopwords'
]

for package in required_packages:
    try:
        nltk.data.find(package)
    except LookupError:
        nltk.download(package, download_dir=nltk_data_path)

# Initialize spell checker
spell = SpellChecker()

# ========== Enhanced NLP Logic ==========
def get_wordnet_pos(treebank_tag):
    """Enhanced POS tag converter with more categories"""
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None  # Return None for other POS tags

def correct_spelling(text):
    """More robust spelling correction"""
    tokens = word_tokenize(text)
    corrected = []
    for word in tokens:
        if word.lower() not in spell:
            # Try to correct but keep original if no good suggestion
            suggestion = spell.correction(word)
            if suggestion and len(suggestion) > 0:
                corrected.append(suggestion)
            else:
                corrected.append(word)
        else:
            corrected.append(word)
    return ' '.join(corrected)

def get_sense_definition(word, sense_type):
    """Get predefined definitions for special cases"""
    definitions = {
        'bat': {
            'sports': 'a club used for hitting the ball in sports like baseball or cricket',
            'animal': 'a flying mammal with wings'
        },
        'bank': {
            'financial': 'a financial institution that handles money',
            'river': 'the land alongside a body of water'
        },
        'book': {
            'reading': 'a written or printed work consisting of pages',
            'reserve': 'to arrange for something in advance'
        }
    }
    return definitions.get(word, {}).get(sense_type, None)

def disambiguate_word(word, context, pos=None):
    """Enhanced word sense disambiguation"""
    # Special cases first
    special_cases = {
        'bat': {
            'sports': ['baseball', 'cricket', 'hit', 'game', 'play', 'sport'],
            'animal': ['fly', 'wing', 'mammal', 'night', 'cave', 'flying']
        },
        'bank': {
            'financial': ['money', 'account', 'loan', 'deposit', 'cash', 'withdraw'],
            'river': ['water', 'fish', 'slope', 'side', 'river', 'stream']
        },
        'book': {
            'reading': ['read', 'page', 'chapter', 'novel', 'author'],
            'reserve': ['reservation', 'ticket', 'appointment', 'schedule']
        }
    }
    
    word_lower = word.lower()
    context_words = set(w.lower() for w in word_tokenize(context))
    
    # Check special cases first
    if word_lower in special_cases:
        for sense, triggers in special_cases[word_lower].items():
            if any(trigger in context_words for trigger in triggers):
                return get_sense_definition(word_lower, sense)
    
    # Try WordNet lookup
    synsets = wn.synsets(word, pos=pos)
    if synsets:
        # Get the most common sense
        return synsets[0].definition()
    
    return None

def process_input(text):
    """Enhanced processing pipeline"""
    corrected = correct_spelling(text)
    tokens = word_tokenize(corrected)
    tags = pos_tag(tokens)
    senses = {}
    
    for word, tag in tags:
        wn_pos = get_wordnet_pos(tag)
        definition = disambiguate_word(word, corrected, wn_pos)
        if definition:
            senses[word] = definition
    
    return corrected, tags, senses

def generate_response(corrected, pos_tags, senses):
    """More contextual response generation"""
    lowered = corrected.lower()
    
    # Handle special cases
    if 'bat' in senses:
        if 'sports' in senses['bat']:
            return "Are you talking about baseball or cricket? 🏏"
        else:
            return "Interesting! Bats are the only flying mammals. 🦇"
    
    if 'bank' in senses:
        if 'financial' in senses['bank']:
            return "Talking about money matters? 💰"
        else:
            return "Ah, the peaceful riverbank! 🌊"
    
    if 'book' in senses:
        if 'reading' in senses['book']:
            return "Books open doors to new worlds! What are you reading? 📚"
        else:
            return "Making a reservation? 🗓️"
    
    if 'love' in lowered:
        return "Love makes the world go round! ❤️"
    
    # Default response
    interesting_words = [word for word in senses if word.lower() not in ['i', 'you', 'the', 'a']]
    if interesting_words:
        return f"Interesting! Tell me more about {interesting_words[0]}."
    return "Thanks for sharing! What else would you like to discuss?"

# ========== Beautiful UI Components ==========
def display_pos_tags(tags):
    """Visualize POS tags with color coding"""
    pos_colors = {
        'NOUN': '#4CC9F0', 
        'VERB': '#F72585',
        'ADJ': '#7209B7',
        'ADV': '#3A0CA3'
    }
    
    # Simplify tags
    simplified_tags = []
    for word, tag in tags:
        simple_tag = 'NOUN' if tag.startswith('NN') else \
                   'VERB' if tag.startswith('VB') else \
                   'ADJ' if tag.startswith('JJ') else \
                   'ADV' if tag.startswith('RB') else 'OTHER'
        simplified_tags.append((word, tag, simple_tag))
    
    # Create colored tags
    tags_html = "<div style='line-height: 2.5;'>"
    for word, original_tag, simple_tag in simplified_tags:
        color = pos_colors.get(simple_tag, '#4361EE')
        tags_html += f"""
        <span style='display: inline-block; margin: 0.1em; padding: 0.3em 0.6em; 
        border-radius: 0.5em; background-color: {color}; color: white; font-size: 0.9em;'>
        {word} <small>({original_tag})</small>
        </span>
        """
    tags_html += "</div>"
    st.markdown(tags_html, unsafe_allow_html=True)

def display_word_senses(senses):
    """Visualize word senses with expandable sections"""
    if not senses:
        st.info("No specific word senses detected in common vocabulary")
        return
    
    for word, definition in senses.items():
        with st.expander(f"🔍 {word.capitalize()}", expanded=True):
            st.markdown(f"**Definition:** {definition}")
            st.progress(80 if len(definition) > 30 else 50)  # Fake confidence indicator

# ========== Streamlit UI ==========
st.set_page_config(
    page_title="Enhanced NLP Bot", 
    page_icon="🤖",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .header {
        color: #6a3093;
        padding: 10px;
        border-bottom: 2px solid #6a3093;
        margin-bottom: 20px;
    }
    .response-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .st-emotion-cache-1y4p8pa {
        padding: 2rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="header"><h1>🤖 Enhanced NLP ContextBot</h1></div>', unsafe_allow_html=True)
st.markdown("""
This bot performs **spelling correction**, **POS tagging**, and **word sense disambiguation** with enhanced 
handling of ambiguous words like "bat" and "bank".
""")

# User input
user_input = st.text_input("You:", key="input", 
                         placeholder="Try: 'I like playing with bat' or 'I went to the bank'...")

if user_input:
    if user_input.lower() == 'exit':
        st.success("Goodbye! Refresh the page to start over.")
    else:
        with st.spinner("Analyzing your input..."):
            time.sleep(0.5)  # Simulate processing time
            
            # Process input
            corrected, pos_tags, senses = process_input(user_input)
            response = generate_response(corrected, pos_tags, senses)
            
            # Display results in columns
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Basic Results")
                st.markdown(f"**Corrected Text:**\n\n`{corrected}`")
                
                st.subheader("Bot Response")
                st.markdown(f'<div class="response-box">{response}</div>', unsafe_allow_html=True)
                
                # Show appropriate emoji based on response
                if '🏏' in response:
                    st.markdown("<h2 style='text-align: center;'>🏏</h2>", unsafe_allow_html=True)
                elif '🦇' in response:
                    st.markdown("<h2 style='text-align: center;'>🦇</h2>", unsafe_allow_html=True)
                elif '💰' in response:
                    st.markdown("<h2 style='text-align: center;'>💰</h2>", unsafe_allow_html=True)
                elif '🌊' in response:
                    st.markdown("<h2 style='text-align: center;'>🌊</h2>", unsafe_allow_html=True)
                elif '📚' in response:
                    st.markdown("<h2 style='text-align: center;'>📚</h2>", unsafe_allow_html=True)
                elif '❤️' in response:
                    st.markdown("<h2 style='text-align: center;'>❤️</h2>", unsafe_allow_html=True)
            
            with col2:
                st.subheader("Detailed Analysis")
                tab1, tab2 = st.tabs(["POS Tags", "Word Senses"])
                
                with tab1:
                    st.markdown("**Color-coded Part-of-Speech Tags**")
                    display_pos_tags(pos_tags)
                    
                    # POS statistics
                    pos_counts = defaultdict(int)
                    for _, tag in pos_tags:
                        simple_tag = 'NOUN' if tag.startswith('NN') else \
                                   'VERB' if tag.startswith('VB') else \
                                   'ADJ' if tag.startswith('JJ') else \
                                   'ADV' if tag.startswith('RB') else 'OTHER'
                        pos_counts[simple_tag] += 1
                    
                    st.markdown("**POS Distribution**")
                    st.bar_chart(pos_counts)
                
                with tab2:
                    st.markdown("**Disambiguated Word Senses**")
                    display_word_senses(senses)
            
            # Success effect
            st.balloons()