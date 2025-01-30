import os.path
import re
from collections import defaultdict, Counter
import pickle
from transformers import PreTrainedTokenizerFast
import os


def load_pretrained_fast(model_path=None):
    if model_path is None:
        model_path = os.path.join(os.getcwd(), "models", "pretrainedModels", "pretrained_tokenizer")
    return PreTrainedTokenizerFast(
        tokenizer_file=model_path,
        eos_token='<[EOS]>',
        pad_token='<[EOS]>'
    )


def compute_frequencies(file_path):
    word_freq = Counter()
    try:
        with open(file_path, 'r', encoding='utf-8') as de_file:
            for de_line in de_file:
                tokens = pre_tokenize(de_line.strip())
                word_freq.update(tokens)

    except FileNotFoundError:
        print(f"File not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return word_freq


def compute_best_pair(pair_freq):
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freq.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    return best_pair


def pre_tokenize(sentence):
    word_list = re.split(pattern=r"(\s\w*|\W)", string=sentence)
    filtered_word_list = list(filter(lambda x: x.strip(), word_list))
    return filtered_word_list


def compute_pair_freq(word_freq, splits):
    pair_freq = defaultdict(int)
    for word, freq in word_freq.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freq[pair] += freq
    return pair_freq


def merge_pair(word_freq, a, b, splits):
    for word in word_freq:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2:]
            else:
                i += 1
        splits[word] = split
    return splits


def create_splits(word_list):
    return {word: [c for c in word] for word in word_list}


def create_alphabet_from_list(word_freq):
    alphabet = []
    for word in word_freq:
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
    alphabet.sort()
    return alphabet


def create_alphabet(word_freq):
    alphabet = []
    for word in word_freq.keys():
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
    alphabet.sort()
    return alphabet


def train_tokenizer(alphabet, word_freq, vocab_size):
    if len(word_freq) < vocab_size:
        vocab_size = word_freq

    merges = {}
    splits = create_splits(word_freq.keys())
    while len(alphabet) < vocab_size:
        pair_freq = compute_pair_freq(word_freq, splits)
        best_pair = compute_best_pair(pair_freq)
        splits = merge_pair(word_freq, *best_pair, splits)
        merges[best_pair] = best_pair[0] + best_pair[1]
        alphabet.append(best_pair[0] + best_pair[1])

    return alphabet, merges


def load_pretrained(load_path):
    with open(load_path, 'rb') as file:
        vocab_encode, vocab_decode, special_tokens, merges = pickle.load(file=file)
    return vocab_encode, vocab_decode, special_tokens, merges


def create_vocab(special_tokens, alphabet):
    vocab_decode = special_tokens + alphabet
    vocab_encode = {token: idx for idx, token in enumerate(vocab_decode)}
    return vocab_encode, vocab_decode


class BytePairTokenizer:

    def __init__(self, vocab_encode, vocab_decode, special_tokens, merges):
        self.vocab_encode = vocab_encode
        self.vocab_decode = vocab_decode
        self.unk, self.start, self.end, self.pad = special_tokens
        self.merges = merges

    @classmethod
    def get_from_pretrained(cls, load_path):
        vocab_encode, vocab_decode, special_tokens, merges = load_pretrained(load_path=load_path)
        tokenizer = cls(vocab_encode=vocab_encode, vocab_decode=vocab_decode, special_tokens=special_tokens,
                        merges=merges)
        return tokenizer

    @classmethod
    def get_from_file(cls, file_path, vocab_size, special_tokens, store_path=None):
        word_freq = compute_frequencies(file_path)
        alphabet = create_alphabet(word_freq)
        alphabet, merges = train_tokenizer(alphabet, word_freq, vocab_size)
        vocab_encode, vocab_decode = create_vocab(special_tokens, alphabet)
        tokenizer = cls(vocab_encode=vocab_encode, vocab_decode=vocab_decode, special_tokens=special_tokens,
                        merges=merges)
        if store_path is not None:
            tokenizer.store_tokenizer(store_path)
        return tokenizer

    def tokenize(self, sentence):
        pre_tokenized_text = pre_tokenize(sentence)
        splits = [[c for c in word] for word in pre_tokenized_text]
        for pair, merge in self.merges.items():
            for idx, split in enumerate(splits):
                i = 0
                while i < len(split) - 1:
                    if split[i] == pair[0] and split[i + 1] == pair[1]:
                        split = split[:i] + [merge] + split[i + 2:]
                    else:
                        i += 1
                splits[idx] = split

        return [self.start] + sum(splits, []) + [self.end]

    def encode(self, sentence):
        token_list = self.tokenize(sentence)
        token_encoded = [self.vocab_encode.get(token, 0) for token in token_list]
        return token_encoded

    def id_from_token(self, token_list):
        token_encoded = [self.vocab_encode.get(token, 0) for token in token_list]
        return token_encoded

    def token_from_id(self, id_list):
        token_list = [self.vocab_decode[idx] for idx in id_list]
        return token_list

    def decode(self, id_list):
        token_list = [self.vocab_decode[idx] for idx in id_list]
        token_decoded = "".join(token_list)
        return token_decoded

    def store_tokenizer(self, store_path):
        with open(store_path, 'wb') as f:
            pickle.dump([self.vocab_encode, self.vocab_decode, [self.unk, self.start, self.end, self.pad], self.merges],
                        f)
