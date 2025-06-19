import spacy
import re
from typing import List, Tuple, Dict, Any

from base_classes.nlp import AbstractNLPModel
from configuration.nlp_configuration import NLPConfiguration

class SpacyNLP(AbstractNLPModel):
    def __init__(self, nlp_config: NLPConfiguration):
        super().__init__(nlp_config)
        self.nlp = spacy.load(nlp_config.model_name)

    def tokenize(self, text) -> List[str]:
        """Tokenize the input text into words or sentences."""
        doc = self.nlp(text)
        return [token.text for token in doc]

    def remove_stopwords(self, text) -> List[str]:
        """Remove stopwords from the tokenized text."""
        tokens = self.tokenize(text)
        return [token for token in tokens if not self.nlp.vocab[token].is_stop]

    def stem(self, text) -> List[str]:
        """Perform stemming on the tokenized text."""
        tokens = self.tokenize(text)
        return [token.lemma_ for token in self.nlp(' '.join(tokens))]  # Using lemmas as a placeholder for stemming

    def lemmatize(self, text) -> List[str]:
        """Perform lemmatization on the tokenized text."""
        tokens = self.tokenize(text)
        return [token.lemma_ for token in self.nlp(' '.join(tokens))]

    def vectorize(self, text) -> List[float]:
        """Convert the processed text into numerical vectors."""
        doc = self.nlp(text)
        return doc.vector.tolist()

    def summarize(self, text) -> str:
        """Summarize the input text."""
        doc = self.nlp(text)
        return ' '.join([sent.text for sent in doc.sents][:1])  # Simple summary with the first sentence

    def pos_tagging(self, text) -> List[Tuple[str, str]]:
        """Perform Part-of-Speech tagging on the tokenized text."""
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]

    def word_segmentation(self, text) -> List[str]:
        """Segment text into words, especially useful for languages without spaces."""
        return self.tokenize(text)  # spaCy handles word segmentation automatically

    def semantic_parsing(self, text) -> Dict[str, Any]:
        """Analyze the meaning of the text."""
        # Placeholder for semantic parsing logic
        return {"entities": [(ent.text, ent.label_) for ent in self.nlp(text).ents]}

    def syntactic_parsing(self, text) -> Dict[str, Any]:
        """Parse the syntactic structure of the text."""
        doc = self.nlp(text)
        return {"dependencies": [(token.text, token.dep_, token.head.text) for token in doc]}

    def lexical_parsing(self, text) -> Dict[str, Any]:
        """Analyze the lexical elements of the text."""
        doc = self.nlp(text)
        return {"lexicon": {token.text: token.lemma_ for token in doc}}

    def extract_nouns(self, text: str) -> List[str]:
        """
        Extract nouns from text using POS tagging, excluding pronouns.
        
        Args:
            text (str): The text to extract nouns from.
            
        Returns:
            List[str]: A list of extracted nouns.
        """
        pos_tags = self.pos_tagging(text)
        nouns = []
        
        for token, pos in pos_tags:
            # spaCy POS tags: NOUN = noun, PROPN = proper noun, PRON = pronoun
            if pos in ['NOUN', 'PROPN']:
                # Convert to lowercase and clean the token
                clean_token = token.strip().lower()
                # Filter out very short tokens and common pronouns that might slip through
                if len(clean_token) > 2 and clean_token not in ['i', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']:
                    # Normalize plural forms to singular
                    normalized_token = self.normalize_plural(clean_token)
                    nouns.append(normalized_token)
        
        return nouns

    def _extract_uuids(self, text: str) -> List[str]:
        """
        Extract UUIDs from text using regex pattern matching.
        
        Args:
            text (str): The text to extract UUIDs from.
            
        Returns:
            List[str]: A list of extracted UUIDs.
        """
        uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
        return re.findall(uuid_pattern, text.lower())

    def _extract_function_names(self, text: str) -> List[str]:
        """
        Extract Python function names from text using regex pattern matching.
        
        Args:
            text (str): The text to extract function names from.
            
        Returns:
            List[str]: A list of extracted function names.
        """
        function_pattern = r'(\w+)\s*\('
        function_names = re.findall(function_pattern, text)
        
        # Clean and filter function names
        clean_functions = []
        for func_name in function_names:
            clean_func = func_name.strip().lower() + "()"
            if len(clean_func) > 2:
                clean_functions.append(clean_func)
        
        return clean_functions

    def normalize_plural(self, word: str) -> str:
        """
        Normalize plural forms to singular using spaCy lemmatization.
        
        Args:
            word (str): The word to normalize.
            
        Returns:
            str: The normalized (singular) form of the word.
        """
        # Use spaCy's lemmatization for better accuracy
        doc = self.nlp(word)
        if doc:
            return doc[0].lemma_.lower()
        
        # Fallback to simple rules if spaCy lemmatization fails
        if word.endswith('ies'):
            return word[:-3] + 'y'
        elif word.endswith('s'):
            return word[:-1]
        else:
            return word

    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract comprehensive keywords from text including nouns, UUIDs, and function names.
        
        Args:
            text (str): The text to extract keywords from.
            
        Returns:
            List[str]: A list of extracted keywords.
        """
        # Extract full UUIDs
        uuids = self._extract_uuids(text)
        uuid_set = set(uuids)
        uuid_sub_tokens = set()
        for uuid_str in uuids:
            parts = uuid_str.split('-')
            for part in parts:
                if len(part) > 0:
                    uuid_sub_tokens.add(part)
        
        # Mask all UUIDs in the text before extracting nouns
        masked_text = text
        for uuid_str in uuids:
            masked_text = masked_text.replace(uuid_str, '')
        
        function_names = self._extract_function_names(masked_text)
        nouns = self.extract_nouns(masked_text)
        
        # Filter out any function name or noun that is a full UUID or a UUID sub-token
        filtered_function_names = [fn for fn in function_names if fn not in uuid_set and fn not in uuid_sub_tokens]
        filtered_nouns = [noun for noun in nouns if noun not in uuid_set and noun not in uuid_sub_tokens]
        
        # Combine all extracted keywords
        all_keywords = []
        all_keywords.extend(uuids)  # Only full UUIDs
        all_keywords.extend(filtered_function_names)
        all_keywords.extend(filtered_nouns)
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for keyword in all_keywords:
            if keyword not in seen:
                unique_keywords.append(keyword)
                seen.add(keyword)
        
        return unique_keywords