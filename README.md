python 3.8.1
install all dependencies via
```pip install -r requirements.txt```

To run the model you need to run
```python main.py --input input.txt --output x.txt --phrases phrases.txt --phrase_extraction_method rake --threshold 0.5```

Where:
1) --input - path to input file
2) --output - path to output file
3) --phrases - path to phrases list
4) --phrase_extraction_method - method to extract phrases from the input file. Method: rake, yake, brute
5) --threshold - threshold for cosine simularity