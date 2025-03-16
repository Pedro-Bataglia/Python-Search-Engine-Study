import numpy
docs = [
    '''About us. We deliver Artificial Intelligence & Machine Learning
       solutions to solve business challenges.''',
    '''Contact information. Email [martin davtyan at filament dot ai]
       if you have any questions''',
    '''Filament Chat. A framework for building and maintaining a scalable
       chatbot capability''',
]

import string 

import nltk
from nltk.tokenize import TreebankWordTokenizer

REMOVE_PUNCTUATION_TABLE = str.maketrans({x: None for x in string.punctuation})
TOKENIZER = TreebankWordTokenizer()

example_doc = docs[1]
example_doc_tokenized = TOKENIZER.tokenize(
    example_doc.translate(REMOVE_PUNCTUATION_TABLE)
    )
example_doc_tokenized

['Contact',
 'information',
 'Email',
 'martin',
 'davtyan',
 'at',
 'filement',
 'dot',
 'ai',
 'if',
 'you',
 'have',
 'any',
 'question']

from nltk.stem.porter import PorterStemmer

STEMMER = PorterStemmer()
example_doc_tokenized_and_stemmed = [STEMMER.stem(token) for token
                                     in example_doc_tokenized]
example_doc_tokenized_and_stemmed

['Contact',
 'information',
 'Email',
 'martin',
 'davtyan',
 'at',
 'filement',
 'dot',
 'ai',
 'if',
 'you',
 'have',
 'any',
 'question']


def tokenize_and_stem(s):
    return [STEMMER.stem(t) for t
            in TOKENIZER.tokenize(s.translate(REMOVE_PUNCTUATION_TABLE))]
query = 'contact'
tokenize_and_stem(query)

['content']


from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, stop_words='english')
vectorizer.fit(docs)

vectorizer.vocabulary__
{'ai': 0,
 'ani': 1,
 'artifici': 2,
 'build': 3,
 'busi': 4,
 'capabl': 5,
 'challeng': 6,
 'chat': 7,
 'chatbot': 8,
 'contact': 9,
 'davtyan': 10,
 'deliv': 11,
 'dot': 12,
 'email': 13,
 'filament': 14,
 'framework': 15,
 'inform': 16,
 'intellig': 17,
 'learn': 18,
 'machin': 19,
 'maintain': 20,
 'martin': 21,
 'question': 22,
 'scalabl': 23,
 'solut': 24,
 'solv': 25}

query = 'contact email to chat to martin'
query_vector = vectorizer.transform([query]).todense()
query_vector

matrix = numpy.matrix([[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0.5, 0. , 0. , 0. ,
         0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. ]])

from sklearn.metrics.pairwise import cosine_similarity

doc_vectors = vectorizer.transform(docs)
similarity = cosine_similarity(query_vector, doc_vectors)
similarity

array = numpy.array([[0.        , 0.48466849, 0.18162735]])

ranks = (-similarity).argsort(axis=None)
ranks

array2 = numpy.array([1, 2, 0])

most_relevant_doc = docs[ranks[0]]
most_relevant_doc

'Content information. Email [martin davtyan at filement dot ai] if you have any questions'

