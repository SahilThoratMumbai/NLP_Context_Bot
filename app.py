import streamlit as st
import re
from collections import defaultdict

# ========== Self-Contained NLP Implementation ==========
class NLPAnalyzer:
    def __init__(self):
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
        
        # Common words for basic spell checking
        self.common_words = {
            'hello', 'hi', 'bank', 'book', 'love', 'river',
            'money', 'financial', 'water', 'read', 'animal'
        }

    def tokenize(self, text):
        """Simple whitespace tokenizer with punctuation handling"""
        return re.findall(r"\w+(?:'\w+)?|\S", text)

    def pos_tag(self, tokens):
        """Simplified POS tagging using word endings and common patterns"""
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

    def get_senses(self, word, context):
        """Custom word sense disambiguation"""
        word = word.lower()
        context_words = set(w.lower() for w in self.tokenize(context))
        
        if word in self.word_senses:
            senses = self.word_senses[word]
            for sense, definition in senses.items():
                if sense in context_words:
                    return {word: definition}
            
            # Default to first sense if no context match
            return {word: next(iter(senses.values()))}
        
        return {}

    def analyze(self, text):
        """Complete text analysis pipeline"""
        tokens = self.tokenize(text)
        tags = self.pos_tag(tokens)
        senses = {}
        
        for token, tag in tags:
            senses.update(self.get_senses(token, text))
        
        return {
            "tokens": tokens,
            "pos_tags": [(token, self.pos_tags.get(tag, 'NOUN')) for token, tag in tags],
            "word_senses": senses
        }

# ========== Streamlit Application ==========
def main():
    st.set_page_config(page_title="NLP ContextBot", page_icon="🧠")
    st.title("🧠 NLP ContextBot")
    st.markdown("""
    This self-contained version uses custom NLP processing without external dependencies.
    It demonstrates word sense disambiguation for a limited vocabulary.
    """)
    
    analyzer = NLPAnalyzer()
    
    user_input = st.text_input("Enter your text (try words like 'bank', 'book', or 'bat'):")
    
    if user_input:
        if user_input.lower() == 'exit':
            st.success("Goodbye! Refresh the page to start over.")
        else:
            with st.spinner("Analyzing..."):
                results = analyzer.analyze(user_input)
                
                st.subheader("Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Tokens:**", results["tokens"])
                    st.write("**POS Tags:**", results["pos_tags"])
                
                with col2:
                    st.write("**Word Senses:**")
                    if results["word_senses"]:
                        for word, sense in results["word_senses"].items():
                            st.write(f"- {word}: {sense}")
                    else:
                        st.write("No known word senses detected")
                
                # Context-aware response
                if "bank" in results["word_senses"]:
                    if "financial" in results["word_senses"]["bank"]:
                        st.success("I see you're talking about money matters!")
                    else:
                        st.success("Ah, the riverbank - nature is beautiful!")
                elif any(word in results["word_senses"] for word in ["book", "bat"]):
                    st.success("Interesting choice of words!")

if __name__ == "__main__":
    main()