import sys


def run(text_file, phrases_file):
    with open(text_file) as f:
        text = f.read()
    with open(phrases_file) as f:
        phrases = f.readlines()
        phrases = [phrase.strip() for phrase in phrases]
    
    print(len(phrases))
    print(len(text))
    


if __name__ == "__main__":
    args = sys.argv[1:]
    text_filename = args[0]
    phrases_filename = args[1]
    run(text_filename, phrases_filename)