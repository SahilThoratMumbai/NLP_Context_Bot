import streamlit as st
import nltk
import re
import os
from collections import defaultdict
from spellchecker import SpellChecker

# ========== Configuration ==========
class HybridNLP:
    def __init__(self):
        # Initialize both NLTK and fallback systems
        self.use_nltk = True
        self.setup_nltk()
        self.setup_fallback()
        
    def setup_nltk(self):
        """Configure NLTK with fallback if resources aren't available"""
        try:
            # Set custom NLTK path
            NLTK_DATA_PATH = os.path.join(os.getcwd(), "nltk_data")
            os.makedirs(NLTK_DATA_PATH, exist_ok=True)
            nltk.data.path.append(NLTK_DATA_PATH)
            
            # Try to load required resources
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
            nltk.word_tokenize(test_text)
            nltk.pos_tag(nltk.word_tokenize(test_text))
            
        except Exception as e:
            self.use_nltk = False
            st.warning(f"Falling back to simplified NLP: {str(e)}")
    
    def setup_fallback(self):
        """Initialize the self-contained fallback system"""
        # Custom word senses database
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
    
    # ========== NLP Methods ==========
    def tokenize(self, text):
        """Tokenization with fallback"""
        if self.use_nltk:
            try:
                return nltk.word_tokenize(text)
            except:
                self.use_nltk = False
                return self.fallback_tokenize(text)
        return self.fallback_tokenize(text)
    
    def fallback_tokenize(self, text):
        """Simple whitespace tokenizer with punctuation handling"""
        return re.findall(r"\w+(?:'\w+)?|\S", text)
    
    def pos_tag(self, tokens):
        """POS tagging with fallback"""
        if self.use_nltk:
            try:
                return nltk.pos_tag(tokens)
            except:
                self.use_nltk = False
                return self.fallback_pos_tag(tokens)
        return self.fallback_pos_tag(tokens)
    
    def fallback_pos_tag(self, tokens):
        """Simplified POS tagging using word endings"""
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
            return text  # Return original if correction fails
    
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
    
    def get_senses(self, word, context):
        """Word sense disambiguation with fallback"""
        word = word.lower()
        context_words = set(w.lower() for w in self.tokenize(context))
        
        # First try NLTK WordNet if available
        if self.use_nltk:
            try:
                tags = self.pos_tag([word])
                wn_pos = self.get_wordnet_pos(tags[0][1])
                
                best_sense = None
                max_overlap = 0
                for sense in wn.synsets(word, pos=wn_pos):
                    signature = set(self.tokenize(sense.definition()))
                    overlap = len(context_words.intersection(signature))
                    if overlap > max_overlap:
                        best_sense = sense
                        max_overlap = overlap
                
                if best_sense:
                    return {word: best_sense.definition()}
            except:
                self.use_nltk = False
        
        # Fallback to custom senses
        if word in self.word_senses:
            senses = self.word_senses[word]
            for sense, definition in senses.items():
                if sense in context_words:
                    return {word: definition}
            return {word: next(iter(senses.values()))}
        
        return {}
    
    def analyze(self, text):
        """Complete text analysis pipeline"""
        corrected = self.correct_spelling(text)
        tokens = self.tokenize(corrected)
        tags = self.pos_tag(tokens)
        senses = {}
        
        for token, tag in tags:
            senses.update(self.get_senses(token, corrected))
        
        return {
            "corrected": corrected,
            "tokens": tokens,
            "pos_tags": tags,
            "word_senses": senses
        }

# ========== Streamlit UI ==========
def main():
    st.set_page_config(page_title="Hybrid NLP Bot", page_icon="🤖")
    st.title("🤖 Hybrid NLP ContextBot")
    st.markdown("""
    This bot combines NLTK functionality with fallback methods when resources aren't available.
    It performs spelling correction, POS tagging, and word sense disambiguation.
    """)
    
    analyzer = HybridNLP()
    
    user_input = st.text_input("You:", key="input", 
                             placeholder="Try words like 'bank', 'book', or 'bat'...")
    
    if user_input:
        if user_input.lower() == "exit":
            st.markdown("### 👋 Bot: Goodbye!")
        else:
            with st.spinner("Analyzing..."):
                try:
                    results = analyzer.analyze(user_input)
                    
                    st.subheader("Analysis Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Corrected Text:**", results["corrected"])
                        st.write("**Tokens:**", results["tokens"])
                    
                    with col2:
                        st.write("**POS Tags:**", results["pos_tags"])
                        st.write("**Word Senses:**")
                        if results["word_senses"]:
                            for word, sense in results["word_senses"].items():
                                st.write(f"- {word}: {sense}")
                        else:
                            st.write("No specific senses detected")
                    
                    # Generate response
                    if "bank" in results["word_senses"]:
                        if "financial" in results["word_senses"]["bank"]:
                            st.success("💰 Talking about banking and finances?")
                        else:
                            st.success("🌊 Ah, the peaceful riverbank!")
                    elif "book" in results["word_senses"]:
                        st.success("📚 Books are wonderful, aren't they?")
                    elif "bat" in results["word_senses"]:
                        st.success("🦇 Interesting! Are we discussing animals or sports?")
                    else:
                        st.info("🤔 Tell me more about what you're thinking!")
                
                except Exception as e:
                    st.error(f"⚠️ Error in analysis: {str(e)}")
                    st.info("The bot is using simplified processing for this input.")

if __name__ == "__main__":
    main()