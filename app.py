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

def disambiguate_word(word, context, pos=None):
    """Enhanced word sense disambiguation"""
    # Special cases first
    special_cases = {
        'bat': {
            'sports': ['baseball', 'cricket', 'hit', 'game', 'play'],
            'animal': ['fly', 'wing', 'mammal', 'night', 'cave']
        },
        'bank': {
            'financial': ['money', 'account', 'loan', 'deposit'],
            'river': ['water', 'fish', 'slope', 'side']
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

def get_sense_definition(word, sense_type):
    """Get predefined definitions for special cases"""
    definitions =