from typing import List, Dict
import string

import nltk
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from nltk.util import ngrams 
from nltk.corpus import stopwords


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english')) 
punkts = set([*string.punctuation])


class Suggestor():
    
    BRUTE_PHRASE_EXTRACTION = "brute"
    
    def __init__(self, comporator, phrase_extraction_strategy: str = BRUTE_PHRASE_EXTRACTION) -> None:
        self.comporator = comporator
        self.phrase_extraction = phrase_extraction_strategy
        
    
    def get_suggestions(self, text: str, phrases: List[str], window: int = 1) -> Dict[str, str]:
        
        if self.phrase_extraction == Suggestor.BRUTE_PHRASE_EXTRACTION:
            text_phrases = self.brute_phrase_extraction(text, window=window)
        else:
            raise Exception("Phrase extraction strategy is not set")
        
        return self.comporator.compare_phrases(text_phrases, phrases)
    
    
    def brute_phrase_extraction(self, text: str, window: int) -> List:
        sentences = sent_tokenize(text)
        sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
        sentences = [[word for word in sentence if word not in punkts] for sentence in sentences]
        sentences = [[word for word in sentence if word not in stop_words] for sentence in sentences]
        
        phrases = []
        for sentence in sentences:
            n_grams = list(ngrams(sentence, window))
            for n_gram in n_grams:
                phrases.append(" ".join(list(n_gram)))
        
        return phrases
    

class PhraseComporator():
    
    WHALELOOPS_BERT = 'whaleloops/phrase-bert'
    
    def __init__(self, model_name: str, threshold: float) -> None:
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
    
    
    def compare_phrases(self, phrases1: List[str], phrases2: List[str]) -> List:
        left_group = self.model.encode(phrases1)
        right_group = self.model.encode(phrases2)
        
        cov_matrix = cosine_similarity(left_group, right_group)
        mapper = []
        for phrase1_idx, row in enumerate(cov_matrix):
            phrase_2idx = row.argmax()
            score = row[phrase_2idx]
            if score <= self.threshold:
                continue
            mapper.append((phrases1[phrase1_idx], phrases2[phrase_2idx], score))
        return mapper
        
        
        
        
