def unigram_counts(sequences):
    return Counter(sequences)


tags = [tag for i, (word, tag) in enumerate(data.training_set.stream())]
tag_unigrams = unigram_counts(tags)

def bigram_counts(sequences):
    d = Counter(sequences)
    return d

tags = [tag for i, (word, tag) in enumerate(data.stream())]
o = [(tags[i],tags[i+1]) for i in range(0,len(tags)-2,2)]
tag_bigrams = bigram_counts(o)


def starting_counts(sequences):
    d = Counter(sequences)
    return d

tags = [tag for i, (word, tag) in enumerate(data.stream())]
starts_tag = [i[0] for i in data.Y]
tag_starts = starting_counts(starts_tag)


def ending_counts(sequences):
    d = Counter(sequences)
    return d

end_tag = [i[len(i)-1] for i in data.Y]
tag_ends = ending_counts(end_tag)


basic_model = HiddenMarkovModel(name="base-hmm-tagger")

tags = [tag for i, (word, tag) in enumerate(data.stream())]
words = [word for i, (word, tag) in enumerate(data.stream())]

tags_count=unigram_counts(tags)
tag_words_count=pair_counts(tags,words)

starting_tag_list=[i[0] for i in data.Y]
ending_tag_list=[i[-1] for i in data.Y]
starting_tag_count=starting_counts(starting_tag_list)#the number of times a tag occured at the start
ending_tag_count=ending_counts(ending_tag_list) #the number of times a tag occured at the end

to_pass_states = []
for tag, words_dict in tag_words_count.items():
    total = float(sum(words_dict.values()))
    distribution = {word: count/total for word, count in
                    words_dict.items()}
    tag_emissions = DiscreteDistribution(distribution)
    tag_state = State(tag_emissions, name=tag)
    to_pass_states.append(tag_state)

basic_model.add_states()
start_prob={}

for tag in tags:
    start_prob[tag]=starting_tag_count[tag]/tags_count[tag]

for tag_state in to_pass_states :
    basic_model.add_transition(basic_model.start,tag_state,start_prob[next_tag_state.name])

end_prob={}

for tag in tags:
    end_prob[tag]=ending_tag_count[tag]/tags_count[tag]

for tag_state in to_pass_states :
    basic_model.add_transition(tag_state,basic_model.end,end_prob[tag_state.name])

transition_prob_pair={}
for key in tag_bigrams.keys():
    transition_prob_pair[key]=tag_bigrams.get(key)/tags_count[key[0]]
for tag_state in to_pass_states :
    for next_tag_state in to_pass_states :
        basic_model.add_transition(tag_state,next_tag_state,transition_prob_pair[(
            tag_state.name,next_tag_state.name)])
basic_model.bake()
