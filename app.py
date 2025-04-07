import streamlit as st
import nltk
import re
import os
from collections import defaultdict
from spellchecker import SpellChecker
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# ========== Configuration ==========
class EnhancedNLP:
    def __init__(self):
        self.use_nltk = True
        self.setup_resources()
        self.setup_fallback()
        
    def setup_resources(self):
        """Configure NLTK with proper resource handling"""
        try:
            # Set NLTK data path
            NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")
            os.makedirs(NLTK_DATA_PATH, exist_ok=True)
            nltk.data.path.append(NLTK_DATA_PATH)

            # Download required resources
            required_packages = [
                ("punkt", "tokenizers/punkt"),
                ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
                ("wordnet", "corpora/wordnet"),
                ("omw-1.4", "corpora/omw-1.4")
            ]
            
            for package, path in required_packages:
                try:
                    nltk.data.find(path)
                except LookupError:
                    nltk.download(package, download_dir=NLTK_DATA_PATH)
            
            # Test NLTK functionality
            test_text = "Testing NLTK"
            word_tokenize(test_text)
            pos_tag(word_tokenize(test_text))
            
        except Exception as e:
            self.use_nltk = False
            st.warning(f"Using simplified NLP: {str(e)}")
    
    def setup_fallback(self):
        """Initialize fallback systems"""
        # Custom word senses
        self.word_senses = {
            "bank": {
                "financial": "a financial institution that accepts deposits",
                "river": "sloping land beside a body of water"
            },
            "book": {
                "reading": "a written or printed work",
                "reserve": "to arrange for something in advance"
            },
            "bat": {
                "animal": "a flying mammal",
                "sports": "a club used in baseball"
            }
        }
        
        # POS tag mapping
        self.pos_tags = {
            'NN': 'NOUN', 'VB': 'VERB', 'JJ': 'ADJ', 'RB': 'ADV',
            'NNS': 'NOUN', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB',
            'VBP': 'VERB', 'VBZ': 'VERB', 'JJR': 'ADJ', 'JJS': 'ADJ',
            'RBR': 'ADV', 'RBS': 'ADV'
        }
        
        # Initialize spell checker
        self.spell = SpellChecker()
    
    # ========== Core NLP Methods ==========
    def tokenize(self, text):
        """Tokenization with fallback"""
        if self.use_nltk:
            try:
                return word_tokenize(text)
            except:
                self.use_nltk = False
                return self.fallback_tokenize(text)
        return self.fallback_tokenize(text)
    
    def fallback_tokenize(self, text):
        """Simple regex tokenizer"""
        return re.findall(r"\w+(?:'\w+)?|\S", text)
    
    def tag_pos(self, tokens):
        """POS tagging with fallback"""
        if self.use_nltk:
            try:
                return pos_tag(tokens)
            except:
                self.use_nltk = False
                return self.fallback_pos_tag(tokens)
        return self.fallback_pos_tag(tokens)
    
    def fallback_pos_tag(self, tokens):
        """Rule-based POS tagging"""
        tags = []
        for token in tokens:
            lower_token = token.lower()
            if lower_token.endswith('ing'):
                tags.append((token, 'VBG'))
            elif lower_token.endswith('ed'):
                tags.append((token, 'VBD'))
            elif lower_token.endswith('ly'):
                tags.append((token, 'RB'))
            elif lower_token.endswith('s'):
                tags.append((token, 'NNS'))
            elif lower_token[0].isupper() and len(token) > 1:
                tags.append((token, 'NNP'))
            elif lower_token in {'is', 'am', 'are', 'was', 'were'}:
                tags.append((token, 'VB'))
            else:
                tags.append((token, 'NN'))
        return tags
    
    def correct_spelling(self, text):
        """Spelling correction with fallback"""
        try:
            tokens = self.tokenize(text)
            corrected = []
            for word in tokens:
                if word.lower() not in self.spell:
                    corrected_word = self.spell.correction(word)
                    corrected.append(corrected_word if corrected_word else word)
                else:
                    corrected.append(word)
            return " ".join(corrected)
        except:
            return text
    
    def get_wordnet_pos(self, treebank_tag):
        """POS to WordNet mapping"""
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
    
    def get_word_senses(self, word, context, pos=None):
        """Hybrid word sense disambiguation"""
        word = word.lower()
        context_words = set(w.lower() for w in self.tokenize(context))
        
        # Try NLTK WordNet first
        if self.use_nltk:
            try:
                wn_pos = self.get_wordnet_pos(pos[1]) if pos else None
                best_sense = None
                max_overlap = 0
                
                for sense in wn.synsets(word, pos=wn_pos):
                    signature = set(self.tokenize(sense.definition()))
                    overlap = len(context_words.intersection(signature))
                    if overlap > max_overlap:
                        best_sense = sense
                        max_overlap = overlap
                
                if best_sense:
                    return best_sense.definition()
            except:
                self.use_nltk = False
        
        # Fallback to custom senses
        if word in self.word_senses:
            senses = self.word_senses[word]
            for sense in senses:
                if sense in context_words:
                    return senses[sense]
            return next(iter(senses.values()))
        
        return None
    
    def process_input(self, text):
        """Complete processing pipeline"""
        corrected = self.correct_spelling(text)
        tokens = self.tokenize(corrected)
        tags = self.tag_pos(tokens)
        senses = {}
        
        for word, tag in tags:
            if word.lower() == "bank" and "river" in [t.lower() for t in tokens]:
                senses[word] = "sloping land beside a body of water"
            else:
                meaning = self.get_word_senses(word, corrected, tag)
                if meaning:
                    senses[word] = meaning
        
        return corrected, tags, senses
    
    def generate_response(self, corrected, pos_tags, senses):
        """Response generation logic"""
        lowered = corrected.lower()
        if "bank" in senses:
            meaning = senses["bank"]
            if "financial" in meaning:
                return "Are you talking about a financial institution?"
            else:
                return "Oh! You mean a river bank. Sounds peaceful."
        elif "book" in lowered:
            return "Books are a great source of knowledge!"
        elif "love" in lowered:
            return "Love is a beautiful emotion. Tell me more!"
        else:
            return "Thanks for sharing! What else would you like to talk about?"

# ========== Streamlit UI ==========
def main():
    st.set_page_config(page_title="Enhanced NLP Bot", page_icon="🤖")
    st.title("🤖 Enhanced NLP ContextBot")
    st.markdown("""
    This enhanced version maintains your original NLTK functionality while 
    adding robust fallback mechanisms when resources aren't available.
    """)
    
    nlp = EnhancedNLP()
    
    user_input = st.text_input("You:", key="input", 
                             placeholder="Try words like 'bank', 'book', or 'love'...")
    
    if user_input:
        if user_input.lower() == "exit":
            st.markdown("### 👋 Bot: Goodbye!")
        else:
            with st.spinner("Analyzing..."):
                try:
                    corrected, pos_tags, senses = nlp.process_input(user_input)
                    response = nlp.generate_response(corrected, pos_tags, senses)
                    
                    st.markdown(f"**🔤 Corrected Input:** `{corrected}`")
                    st.markdown(f"**🔠 POS Tags:** `{pos_tags}`")
                    st.markdown(f"**🧠 Word Senses:** `{senses}`")
                    st.markdown(f"### 🤖 Bot: {response}")
                
                except Exception as e:
                    st.error(f"⚠️ Error: {str(e)}")
                    st.info("The bot is using simplified processing for this input.")

if __name__ == "__main__":
    main()