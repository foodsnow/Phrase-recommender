import sys

from model import Suggestor, PhraseComporator

SIMILARITY_SCORE = 0.5
WINDOW_SIZE = 1

def run(text_file: str, phrases_file: str, output: str):
    with open(text_file) as f:
        text = f.read()
    with open(phrases_file) as f:
        phrases = f.readlines()
        phrases = [phrase.strip() for phrase in phrases]
    
    comporator = PhraseComporator(PhraseComporator.WHALELOOPS_BERT, SIMILARITY_SCORE)
    suggestor = Suggestor(comporator)
    suggestions = suggestor.get_suggestions(text, phrases, window=WINDOW_SIZE)
    
    print(f"[INFO]: {len(suggestions)} suggestions were made")
    
    with open(output, "w+") as f:
        for original, suggestion, score in suggestions:
            f.write(f"{original} -> {suggestion}, {score}\n")
    

if __name__ == "__main__":
    args = sys.argv[1:]
    
    if len(args) != 3:
        print("Arguments are not provided!")
        print("E.g. python main.py {input_file} {phrases_file} {output_file}")
        sys.exit(0)
        
    text_filename = args[0]
    phrases_filename = args[1]
    output_filename = args[2]
    run(text_filename, phrases_filename, output_filename)