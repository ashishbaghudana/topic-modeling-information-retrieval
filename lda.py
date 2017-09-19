import codecs
import ftfy
import logging
import pickle
import re
import string
import time

from gensim import corpora
from gensim.models.ldamodel import LdaModel
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def initialize_logging():
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=FORMAT, level=logging.DEBUG)
    logger = logging.getLogger('Dictionary')


class Documents(object):
    def __init__(self, fname):
        self.fname = fname
        self.logger = logging.getLogger('Documents')

    def __iter__(self):
        with codecs.open(self.fname, encoding='utf8') as freader:
            for line in freader:
                yield line


class Text(object):
    def __init__(self, documents, stop_words, tokenization_function, minimum_token_length=3):
        self.documents = documents
        self.stop_words = stop_words
        self.tokenization_function = tokenization_function
        self.minimum_token_length = minimum_token_length
        self.logger = logging.getLogger('Text')

    def __iter__(self):
        for document in self.documents:
            doc = re.sub(r'http\S+', '', ftfy.fix_text(document.lower()))
            doc = re.sub(r'[^\x00-\x7F]+',' ', document)
            yield [
                token
                for token in self.tokenization_function(
                    str(doc).lower().translate(None, string.punctuation))
                if token not in self.stop_words and len(token) > self.minimum_token_length
            ]


class Dictionary(object):
    def __init__(self, corpus=None):
        self.corpus = corpus
        if self.corpus is not None:
            self.documents = Documents(self.corpus)
        else:
            self.documents = None
        self.texts = None
        self.logger = logging.getLogger('LDA-Dictionary')
        self.dictionary = None

    def generate(self,
                 stop_words=stopwords.words('english'),
                 tokenization_function=word_tokenize):
        try:
            assert self.corpus != None and self.documents != None
            self.texts = Text(self.documents, stop_words,
                              tokenization_function)
            self.dictionary = corpora.Dictionary(self.texts)
        except AssertionError:
            self.logger.error('The corpus is a None object')

    def save(self, fname):
        try:
            assert self.dictionary != None
            with open(fname, 'wb') as fwriter:
                pickle.dump(self.dictionary, fwriter)
        except AssertionError:
            self.logger.error('The dictionary object is None')

    def load(self, fname):
        with open(fname, 'rb') as freader:
            self.dictionary = pickle.load(freader)


class Corpus(object):
    def __init__(self,
                 fname,
                 dictionary,
                 tokenization_function=word_tokenize,
                 stop_words=stopwords.words('english')):
        self.fname = fname
        self.dictionary = dictionary
        self.tokenization_function = tokenization_function
        self.stop_words = stop_words

    def __iter__(self):
        with codecs.open(self.fname, encoding='utf8') as freader:
            for line in freader:
                document = re.sub(r'http\S+', '', ftfy.fix_text(line))
                document = re.sub(r'[^\x00-\x7F]+',' ', document)
                yield self.dictionary.doc2bow([
                    token
                    for token in self.tokenization_function(
                        str(document).lower().translate(None, string.punctuation))
                    if token not in self.stop_words and len(token) > 3
                ])


class LDA(object):
    def __init__(self, corpus_file, dictionary):
        self.corpus_file = corpus_file
        self.dictionary = dictionary
        self.model = None
        self.corpus = Corpus(self.corpus_file, self.dictionary)
        self.logger = logging.getLogger('LDA')

    def train(self, num_topics=10):
        self.model = LdaModel(self.corpus, num_topics=num_topics, id2word=self.dictionary)

    def save(self, save_file):
        try:
            assert self.model is not None
            with open(save_file, 'wb') as fwriter:
                pickle.dump(self.model, fwriter)
        except AssertionError:
            self.logger('No model has been created to save')

    def load(self, pickle_file):
        with open(pickle_file, 'rb') as freader:
            self.model = pickle.load(freader)


def generate_dictionary(corpus, save_file):
    dictionary = Dictionary(corpus)
    dictionary.generate()
    dictionary.save(save_file)
    return dictionary.dictionary


def load_dictionary_from_file(pickle_file):
    dictionary = Dictionary()
    dictionary.load(pickle_file)
    return dictionary.dictionary


def main():
    initialize_logging()

    # Generate dictionary
    corpus_file = 'corpus.txt'
    save_file = 'dictionary.bin'
    dictionary = generate_dictionary(corpus_file, save_file)

    # Load dictionary
    # pickle_file = 'dictionary.bin'
    # dictionary = load_dictionary_from_file(pickle_file)

    # Run LDA
    num_topics = 10
    lda = LDA(corpus_file, dictionary)
    lda.train(num_topics)

    # Save model to disk
    model_file = 'lda_model_k_10.bin'
    lda.save(model_file)

    lda.model.print_topics(num_topics)

if __name__ == '__main__':
    main()
