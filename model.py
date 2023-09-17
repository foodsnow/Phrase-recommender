from typing import List, Dict
import string

import nltk
from rake_nltk import Rake
import yake
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
    
    def __init__(self, comporator, phrase_extractor) -> None:
        self.comporator = comporator
        self.phrase_extractor = phrase_extractor
        
    
    def get_suggestions(self, text: str, phrases: List[str]) -> Dict[str, str]:
        text_phrases = self.phrase_extractor.extract(text)
        return self.comporator.compare_phrases_from_to(text_phrases, phrases)


class PhraseExtractor():
    
    BRUTE = "brute"
    RAKE = "rake"
    YAKE = "yake"
    
    def __init__(self, strategy: str, window: int = 1) -> None:
        self.strategy = strategy
        self.window = window
        
    
    def extract(self, text: str):
        if self.strategy == PhraseExtractor.RAKE:
            return self.rake(text)
        elif self.strategy == PhraseExtractor.YAKE:
            return self.yake(text)
        elif self.strategy == PhraseExtractor.BRUTE:
            return self.brute(text, self.window)
        else:
            raise Exception("Phrase extraction strategy is not set")
    
    
    def rake(self, text: str) -> List[str]:
        rake = Rake()
        rake.extract_keywords_from_text(text)
        return rake.get_ranked_phrases()
    
    
    def yake(self, text: str) -> List[str]:
        yake_kw = yake.KeywordExtractor()
        keywords = yake_kw.extract_keywords(text)
        return [phrase[0] for phrase in keywords]
    
    
    def brute(self, text: str, window: int) -> List[str]:
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
    
    
    def compare_phrases_from_to(self, phrases1: List[str], phrases2: List[str]) -> List:
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
        
        
        
        
