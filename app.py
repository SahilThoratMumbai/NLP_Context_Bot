import streamlit as st
import re
from collections import defaultdict

# ========== Self-Contained NLP Implementation ==========
class NLPChatBot:
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
            "love": {
                "emotion": "a strong feeling of affection",
                "score": "zero in tennis"
            }
        }
        
        # POS tag mapping
        self.pos_tags = {
            'NN': 'NOUN', 'VB': 'VERB', 'JJ': 'ADJ', 'RB': 'ADV',
            'NNS': 'NOUN', 'VBD': 'VERB', 'VBG': 'VERB', 'VBN': 'VERB',
            'VBP': 'VERB', 'VBZ': 'VERB', 'JJR': 'ADJ', 'JJS': 'ADJ',
            'RBR': 'ADV', 'RBS': 'ADV'
        }
        
        # Common words dictionary for basic spell checking
        self.dictionary = {
            'hello', 'hi', 'bank', 'book', 'love', 'river',
            'money', 'financial', 'water', 'read', 'knowledge',
            'emotion', 'tell', 'more', 'share', 'thanks'
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

    def correct_spelling(self, text):
        """Basic spelling correction using dictionary lookup"""
        tokens = self.tokenize(text)
        corrected = []
        for word in tokens:
            if word.lower() not in self.dictionary:
                # Very simple correction - just lowercase if not in dictionary
                corrected.append(word.lower())
            else:
                corrected.append(word)
        return ' '.join(corrected)

    def disambiguate_word(self, word, context):
        """Custom word sense disambiguation"""
        word = word.lower()
        context_words = set(w.lower() for w in self.tokenize(context))
        
        if word in self.word_senses:
            senses = self.word_senses[word]
            for sense, definition in senses.items():
                if sense in context_words:
                    return definition
            
            # Default to first sense if no context match
            return next(iter(senses.values()))
        
        return None

    def process_input(self, text):
        """Complete NLP processing pipeline"""
        corrected = self.correct_spelling(text)
        tokens = self.tokenize(corrected)
        tags = self.pos_tag(tokens)
        senses = {}
        
        for token, tag in tags:
            meaning = self.disambiguate_word(token, corrected)
            if meaning:
                senses[token] = meaning
        
        return corrected, tags, senses

    def generate_response(self, corrected, pos_tags, senses):
        """Context-aware response generation"""
        lowered = corrected.lower()
        
        if "bank" in senses:
            if "financial" in senses["bank"]:
                return "Are you talking about a financial institution?"
            else:
                return "Oh! You mean a river bank. Sounds peaceful."
        
        if "book" in lowered:
            return "Books are a great source of knowledge!"
        
        if "love" in lowered:
            return "Love is a beautiful emotion. Tell me more!"
        
        return "Thanks for sharing! What else would you like to talk about?"

# ========== Streamlit UI ==========
def main():
    st.set_page_config(page_title="NLP ContextBot", page_icon="🧠")
    st.title("🧠 NLP ContextBot")
    st.markdown("""
    This self-contained version performs:
    - **Spelling correction** (basic)
    - **POS tagging** (rule-based)
    - **Word sense disambiguation** (custom)
    """)
    
    bot = NLPChatBot()
    
    user_input = st.text_input("You:", key="input",
                             placeholder="Try words like 'bank', 'book', or 'love'...")
    
    if user_input:
        if user_input.lower() == 'exit':
            st.success("Goodbye! Refresh the page to start over.")
        else:
            with st.spinner("Analyzing..."):
                try:
                    corrected, pos_tags, senses = bot.process_input(user_input)
                    response = bot.generate_response(corrected, pos_tags, senses)
                    
                    st.subheader("Analysis Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Corrected Text:**", corrected)
                        st.write("**POS Tags:**", pos_tags)
                    
                    with col2:
                        st.write("**Word Senses:**")
                        if senses:
                            for word, sense in senses.items():
                                st.write(f"- {word}: {sense}")
                        else:
                            st.write("No specific senses detected")
                    
                    st.markdown(f"### 🤖 Bot: {response}")
                
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please try a different input")

if __name__ == "__main__":
    main()