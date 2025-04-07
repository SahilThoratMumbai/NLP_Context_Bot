import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from spellchecker import SpellChecker
import os
from pywsd.lesk import cosine_lesk
from pywsd.utils import lemmatize_sentence

# ========== Setup with Robust Error Handling ==========
class NLPChatBot:
    def __init__(self):
        self.setup_nltk()
        self.spell = SpellChecker()
        self.use_advanced_lesk = True  # Flag for cosine_lesk availability

    def setup_nltk(self):
        """Configure NLTK with multiple fallback options"""
        try:
            # Try multiple possible NLTK data paths
            nltk_data_paths = [
                os.path.join(os.getcwd(), "nltk_data"),
                os.path.join(os.path.expanduser("~"), "nltk_data"),
                "/tmp/nltk_data"
            ]
            
            for path in nltk_data_paths:
                try:
                    os.makedirs(path, exist_ok=True)
                    nltk.data.path.append(path)
                except:
                    continue

            # Download required resources
            required_packages = [
                'punkt', 'averaged_perceptron_tagger',
                'wordnet', 'omw-1.4', 'stopwords'
            ]
            
            for package in required_packages:
                try:
                    nltk.data.find(package)
                except LookupError:
                    nltk.download(package)

        except Exception as e:
            st.warning(f"Limited functionality: {str(e)}")

    def get_wordnet_pos(self, treebank_tag):
        """Convert treebank POS tags to WordNet POS tags"""
        tag_map = {
            'J': wn.ADJ,
            'V': wn.VERB,
            'N': wn.NOUN,
            'R': wn.ADV
        }
        return tag_map.get(treebank_tag[0], wn.NOUN)

    def correct_spelling(self, text):
        """Enhanced spelling correction with error handling"""
        try:
            tokens = word_tokenize(text)
            corrected = []
            for word in tokens:
                if word.lower() not in self.spell:
                    suggestion = self.spell.correction(word)
                    corrected.append(suggestion if suggestion else word)
                else:
                    corrected.append(word)
            return ' '.join(corrected)
        except:
            return text  # Return original if correction fails

    def disambiguate_word(self, word, context, pos=None):
        """Hybrid word sense disambiguation"""
        try:
            if self.use_advanced_lesk:
                try:
                    # Try cosine_lesk first
                    sense = cosine_lesk(context, word, pos=pos)
                    if sense:
                        return sense.definition()
                except:
                    self.use_advanced_lesk = False
                    return self.simple_lesk(word, context, pos)
            
            # Fallback to simple Lesk
            return self.simple_lesk(word, context, pos)
            
        except:
            return None

    def simple_lesk(self, word, context, pos=None):
        """Simplified Lesk algorithm fallback"""
        context_words = set(word_tokenize(context.lower()))
        best_sense = None
        max_overlap = 0
        
        for sense in wn.synsets(word, pos=pos):
            signature = set(word_tokenize(sense.definition().lower()))
            for example in sense.examples():
                signature.update(word_tokenize(example.lower()))
            
            overlap = len(context_words.intersection(signature))
            if overlap > max_overlap:
                max_overlap = overlap
                best_sense = sense
        
        return best_sense.definition() if best_sense else None

    def process_input(self, text):
        """Complete NLP processing pipeline"""
        corrected = self.correct_spelling(text)
        tokens = word_tokenize(corrected)
        pos_tags = pos_tag(tokens)
        senses = {}

        for word, tag in pos_tags:
            wn_pos = self.get_wordnet_pos(tag)
            
            # Special case for "bank"
            if word.lower() == "bank":
                if "river" in [t.lower() for t in tokens]:
                    senses[word] = "sloping land beside a body of water"
                else:
                    senses[word] = "financial institution"
                continue
                
            # General case for other words
            definition = self.disambiguate_word(word, corrected, wn_pos)
            if definition:
                senses[word] = definition
        
        return corrected, pos_tags, senses

    def generate_response(self, corrected, pos_tags, senses):
        """Context-aware response generation"""
        lowered = corrected.lower()
        
        if "bank" in senses:
            meaning = senses["bank"]
            if "financial" in meaning:
                return "Are you talking about banking services?"
            else:
                return "Ah, the riverbank - nature is beautiful!"
        
        if "book" in lowered:
            return "Books are wonderful sources of knowledge!"
        
        if "love" in lowered:
            return "Love is a powerful emotion. Tell me more!"
        
        return "Interesting! What else would you like to discuss?"

# ========== Streamlit UI ==========
def main():
    st.set_page_config(page_title="Enhanced NLP Bot", page_icon="🤖")
    st.title("🤖 Enhanced NLP ContextBot")
    st.markdown("""
    This enhanced version uses:
    - **cosine_lesk** for better word sense disambiguation
    - Robust fallback mechanisms
    - Improved error handling
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
                    st.error(f"Error in processing: {str(e)}")
                    st.info("The bot is using simplified processing for this input.")

if __name__ == "__main__":
    main()