import sys
import argparse

from model import Suggestor, PhraseComporator, PhraseExtractor

SIMILARITY_SCORE_THRESHOLD = 0.5
WINDOW_SIZE = 2

arg_parser = argparse.ArgumentParser(
    prog="Phrase",
    description="lorem ipsum"
)

def run(text_file: str, phrases_file: str, output: str, model_name, phrase_method, threshold):
    with open(text_file) as f:
        text = f.read()
    with open(phrases_file) as f:
        phrases = f.readlines()
        phrases = [phrase.strip() for phrase in phrases]
    
    comporator = PhraseComporator(model_name, threshold)
    extractor = PhraseExtractor(phrase_method, window=WINDOW_SIZE)
    suggestor = Suggestor(comporator, extractor)
    suggestions = suggestor.get_suggestions(text, phrases)
    
    print(f"[INFO]: {len(suggestions)} suggestions were made")
    
    with open(output, "w+") as f:
        for original, suggestion, score in suggestions:
            f.write(f"{original} -> {suggestion}, {score}\n")
    
    
def filter_args(args):
    if args.phrase_extraction_method not in set([ getattr(PhraseExtractor, x) for x in dir(PhraseExtractor) if not x.startswith("__")]):
        raise Exception("not valid")
    if args.model_name not in set([ getattr(PhraseComporator, x) for x in dir(PhraseComporator) if not x.startswith("__")]):
        raise Exception("not valid")
    

if __name__ == "__main__":
    
    arg_parser.add_argument("--input", type=str, required=True)
    arg_parser.add_argument("--phrases", type=str, required=True)
    arg_parser.add_argument("--output", type=str, required=True)
    arg_parser.add_argument("--model_name", type=str, default=PhraseComporator.WHALELOOPS_BERT)
    arg_parser.add_argument("--phrase_extraction_method", type=str, default=PhraseExtractor.RAKE)
    arg_parser.add_argument("--threshold", type=float, default=SIMILARITY_SCORE_THRESHOLD)
    args = arg_parser.parse_args()
    filter_args(args)
        
    text_filename = args.input
    phrases_filename = args.phrases
    output_filename = args.output
    model_name = args.model_name
    phrase_method = args.phrase_extraction_method
    threshold = args.threshold
    run(text_filename, phrases_filename, output_filename, model_name, phrase_method, threshold)