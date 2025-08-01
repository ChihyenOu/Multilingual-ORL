from collections import Counter
from data.SRL import *
import numpy as np


class Vocab(object):
    PAD, UNK = 0, 1

    def __init__(self, word_counter, label_counter, min_occur_count=2):
        self._id2word = ['<pad>', '<unk>']
        self._wordid2freq = [10000, 10000]
        self._id2extword = ['<pad>', '<unk>']
        self._id2label = ['<pad>'] # remove ['<pad>']

        # If n is omitted or None, most_common() returns all elements in the counter
        for word, count in word_counter.most_common():
            if count > min_occur_count:
                self._id2word.append(word)
                self._wordid2freq.append(count) 

        for label, count in label_counter.most_common():
            self._id2label.append(label)
        print("_id2label: ", self._id2label)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._word2id = reverse(self._id2word)
        if len(self._word2id) != len(self._id2word):
            print("serious bug: words dumplicated, please check!")

        ## ADD 
        # reverse_label = lambda x: dict(zip(x, range(1, len(x)+1))) ## ADD 
        self._label2id = reverse(self._id2label) # old
        # self._label2id = reverse_label(self._id2label) # modify
        if len(self._label2id) != len(self._id2label):
            print("serious bug: ner labels dumplicated, please check!")
        
        self._extword2id = reverse(self._id2extword) ## ADD 

        print("Vocab info: #words %d, #labels %d" % (self.vocab_size, self.label_size))

    def load_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword)
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                self._id2extword.append(values[0])
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        #embeddings = embeddings / np.std(embeddings)

        reverse = lambda x: dict(zip(x, range(len(x))))
        self._extword2id = reverse(self._id2extword)

        if len(self._extword2id) != len(self._id2extword):
            print("serious bug: extern words dumplicated, please check!")

        return embeddings

    def create_pretrained_embs(self, embfile):
        embedding_dim = -1
        word_count = 0
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                if word_count < 1:
                    values = line.split()
                    embedding_dim = len(values) - 1
                word_count += 1
        print('Total words: ' + str(word_count) + '\n')
        print('The dim of pretrained embeddings: ' + str(embedding_dim) + '\n')

        index = len(self._id2extword) - word_count
        embeddings = np.zeros((word_count + index, embedding_dim))
        with open(embfile, encoding='utf-8') as f:
            for line in f.readlines():
                values = line.split()
                if self._extword2id.get(values[0], self.UNK) != index:
                    print("Broken vocab or error embedding file, please check!")
                vector = np.array(values[1:], dtype='float64')
                embeddings[self.UNK] += vector
                embeddings[index] = vector
                index += 1

        embeddings[self.UNK] = embeddings[self.UNK] / word_count
        #embeddings = embeddings / np.std(embeddings)

        return embeddings

    def word2id(self, xs):
        if isinstance(xs, list):
            return [self._word2id.get(x, self.UNK) for x in xs]
        return self._word2id.get(xs, self.UNK)

    def id2word(self, xs):
        if isinstance(xs, list):
            return [self._id2word[x] for x in xs]
        return self._id2word[xs]

    def wordid2freq(self, xs):
        if isinstance(xs, list):
            return [self._wordid2freq[x] for x in xs]
        return self._wordid2freq[xs]

    def extword2id(self, xs):
        if isinstance(xs, list):
            return [self._extword2id.get(x, self.UNK) for x in xs]
        return self._extword2id.get(xs, self.UNK)

    def id2extword(self, xs):
        if isinstance(xs, list):
            return [self._id2extword[x] for x in xs]
        return self._id2extword[xs]

    def label2id(self, xs):
        if isinstance(xs, list):
            return [self._label2id.get(x, self.PAD) for x in xs]
        return self._label2id.get(xs, self.PAD)

    def id2label(self, xs):
        if isinstance(xs, list):
            return [self._id2label[x] for x in xs]
        return self._id2label[xs]

    @property
    def vocab_size(self):
        return len(self._id2word)

    @property
    def extvocab_size(self):
        return len(self._id2extword)

    @property
    def label_size(self):
        return len(self._id2label) # +1 # Change


def creat_vocab(corpusFile1, corpusFile2, min_occur_count):
    word_counter = Counter()
    label_counter = Counter()
    with open(corpusFile1, 'r', encoding='utf8') as infile:
        for sentence in readSRL(infile):
            index = 0
            for token in sentence.words:
                word_counter[token.form] += 1
                # if index < sentence.key_start or index > sentence.key_end: # Old version
                if index not in sentence.key_list: # New version
                    label_counter[token.label] += 1
                index = index + 1
    # ADD BELOW lines for second corpus
    with open(corpusFile2, 'r', encoding='utf8') as infile:
        for sentence in readSRL(infile):
            index = 0
            for token in sentence.words:
                word_counter[token.form] += 1
                # if index < sentence.key_start or index > sentence.key_end: # Old version
                if index not in sentence.key_list: # New version
                    label_counter[token.label] += 1
                index = index + 1

    return Vocab(word_counter, label_counter, min_occur_count)
    