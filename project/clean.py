import re
import nltk
import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

def review2wordlist(review, no_stopwords=False):
  review_text = BeautifulSoup(review, "html.parser").get_text()
  review_text = re.sub(r"[^a-zA-Z]", " ", review_text)
  words = review_text.lower().split()
  if no_stopwords:
    words = list(filter(lambda w: not w in stops, words))
  return words

def review2sentences(review, tokenizer, no_stopwords=False):
  raw_sentences = tokenizer.tokenize(review.strip())
  sentences = []
  for raw_sentence in raw_sentences:
    if len(raw_sentence) > 0:
      sentences.append(review2wordlist(raw_sentence, no_stopwords))
  return sentences