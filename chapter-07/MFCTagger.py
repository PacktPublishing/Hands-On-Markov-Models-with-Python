def pair_counts(tags, words):
    d = defaultdict(lambda: defaultdict(int))
    for tag, word in zip(tags, words):
        d[tag][word] += 1
    return d

tags = [tag for i, (word, tag) in enumerate(data.training_set.stream())]
words = [word for i, (word, tag) in enumerate(data.training_set.stream())]

FakeState = namedtuple('FakeState', 'name')


class MFCTagger:
    missing = FakeState(name = '<MISSING>')

    def __init__(self, table):
        self.table = defaultdict(lambda: MFCTagger.missing)
        self.table.update({word: FakeState(name=tag) for word, tag in
                          table.items()})

    def viterbi(self, seq):
    """This method simplifies predictions by matching the Pomegranate
        viterbi() interface"""
        return 0., list(enumerate(["<start>"] + [self.table[w] for w in
                                                 seq] + ["<end>"]))


tags = [tag for i, (word, tag) in enumerate(data.training_set.stream())]
words = [word for i, (word, tag) in enumerate(data.training_set.stream())]

word_counts = pair_counts(words, tags)
mfc_table = dict((word, max(tags.keys(), key=lambda key: tags[key])) for
                  word, tags in word_counts.items())

mfc_model = MFCTagger(mfc_table)


def replace_unknown(sequence):
    return [w if w in data.training_set.vocab else 'nan' for w in sequence]


def simplify_decoding(X, model):
    _, state_path = model.viterbi(replace_unknown(X))
    return [state[1].name for state in state_path[1:-1]]


def accuracy(X, Y, model):
    correct = total_predictions = 0
    for observations, actual_tags in zip(X, Y):
    # The model.viterbi call in simplify_decoding will return None if the HMM
    # raises an error (for example, if a test sentence contains a word that
    # is out of vocabulary for the training set). Any exception counts the
    # full sentence as an error (which makes this a conservative estimate).
        try:
            most_likely_tags = simplify_decoding(observations, model)
            correct += sum(p == t for p, t in zip(most_likely_tags, 
                                                  actual_tags))
        except:
            pass
        total_predictions += len(observations)
    return correct / total_predictions
