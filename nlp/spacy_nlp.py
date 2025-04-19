import spacy
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