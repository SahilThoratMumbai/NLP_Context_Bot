import streamlit as st
import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

class NLPChatBot:
    def __init__(self):
        self.river_context_words = {'river', 'water', 'flow', 'stream', 'sit', 'near', 'by', 'side'}
        self.bank_context_words = {'money', 'account', 'deposit', 'withdraw', 'loan', 'financial'}
        
    def pos_tag(self, text):
        """Improved POS tagging with better verb detection"""
        tokens = nltk.word_tokenize(text)
        pos_tags = nltk.pos_tag(tokens)
        
        # Manual corrections for common mistakes
        corrected_tags = []
        for i, (word, tag) in enumerate(pos_tags):
            # Fix "am" being tagged as NN
            if word.lower() == 'am' and tag == 'NN':
                corrected_tags.append((word, 'VBP'))  # Verb, present tense
            # Fix "near" being tagged as NN when used as preposition
            elif word.lower() == 'near' and tag == 'NN' and i > 0 and pos_tags[i-1][1].startswith('VB'):
                corrected_tags.append((word, 'IN'))  # Preposition
            else:
                corrected_tags.append((word, tag))
        return corrected_tags
    
    def disambiguate_bank(self, sentence):
        """Specialized bank disambiguation"""
        tokens = [word.lower() for word in nltk.word_tokenize(sentence)]
        
        # Count context word matches
        river_score = sum(1 for word in tokens if word in self.river_context_words)
        bank_score = sum(1 for word in tokens if word in self.bank_context_words)
        
        if river_score > bank_score:
            return "🌊 Sloping land beside a body of water (river bank)"
        elif bank_score > river_score:
            return "🏦 Financial institution (money bank)"
        else:
            return "🏦 Financial institution (default)"
    
    def generate_response(self, text, senses):
        """Context-aware response generation"""
        if 'bank' in senses:
            if 'river' in senses['bank'].lower():
                return "🌊 Yes, riverbanks are beautiful places to relax!"
            else:
                return "🏦 Are you discussing financial matters?"
        return "🤔 Interesting! Tell me more."

# Initialize the bot
bot = NLPChatBot()

# Streamlit UI
st.title("🧠 NLP ContextBot Pro")

user_input = st.text_input("💬 You:", "I am sitting near river bank")

if user_input:
    # Process input
    pos_tags = bot.pos_tag(user_input)
    senses = {}
    
    if 'bank' in [word.lower() for word, tag in pos_tags]:
        senses['bank'] = bot.disambiguate_bank(user_input)
    
    response = bot.generate_response(user_input, senses)
    
    # Display results
    st.markdown("### 🔠 POS Tags")
    st.write(" ".join([f"{word} ({tag})" for word, tag in pos_tags]))
    
    if senses:
        st.markdown("### 🧠 Word Senses")
        for word, sense in senses.items():
            st.markdown(f"**{word}**: {sense}")
    
    st.markdown("### 🤖 Response")
    st.write(response)