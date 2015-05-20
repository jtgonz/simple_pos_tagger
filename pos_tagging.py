from __future__ import division
from collections import defaultdict
from itertools import product
import re

def get_unique_unigrams(ngram_counts):
    return {''.join(ngram) for ngram in ngram_counts if len(ngram) == 1 and ngram_counts[ngram]}

def count_unique_unigrams(ngram_counts):
    return sum(ngram_counts[ngram] and len(ngram) == 1 for ngram in ngram_counts) - 1

def count_total_words(emiss_counts):
    return sum(key[0] == '_' and emiss_counts[key] for key in emiss_counts)

def get_all_counts(infile, n=3):
    """
    Iterate through corpus, get n-gram counts and emission counts.
    """

    # create dictionaries to store ngram and emission counts
    ngram_counts = defaultdict(lambda: 0)
    emiss_counts = defaultdict(lambda: 0)

    # initialize tag list to ['*','*','*',...]
    tag_list = ['*'] * n

    with open(infile) as corpus:
        for line in corpus:

            # reached the end of a sentence
            if line == '\n':

                # if there is no data, just skip
                if tag_list[-1] == '*': continue

                word = False
                tag = 'STOP'

            # get word and tag from line
            else:
                word, tag, chunk = line.rstrip().split(' ')

            # add new tag to end of tag list
            tag_list.pop(0)
            tag_list.append(tag)

            # increment count by 1 when ngram appears in corpus
            for i in xrange(1,n+1):
                tag_sequence = tuple(tag_list[-i:])
                ngram_counts[tag_sequence] += 1

            # increment emmision count by 1, or reset tag list if no more words
            if word:
                emiss_counts[tag, word] += 1
                emiss_counts['_', word] += 1
            else:
                tag_list = ['*'] * n

    return ngram_counts, emiss_counts

def get_ml_estimates(ngram_counts, emiss_counts):
    """
    Get maximum likelihood estimates for transition and emission
    probabilities.
    """

    # create dictionaries to store maximum likelihood estimates
    # for transition and emission parameters
    qml_est = defaultdict(lambda: 0)
    eml_est = defaultdict(lambda: 0)

    num_unique_tags = count_unique_unigrams(ngram_counts)
    num_total_words = count_total_words(emiss_counts)

    # get maximum likelihood estimates for transitions
    # qml_est[(u,v,s)] = q_ml(s|u,v) = count(u,v,s) / count(u,v)
    for ngram in ngram_counts.keys():

        if len(ngram) > 1:
            qml_est[ngram] = \
            ngram_counts[ngram] / (ngram_counts[ngram[:-1]] or ngram_counts[('STOP',)])
        else:
            qml_est[ngram] = ngram_counts[ngram] / num_unique_tags

    # get maximum likelihood estimates for emissions
    # eml_est[s][x] = e_ml(x|s) = count(s -> x) / count(s)
    for key in emiss_counts:
        eml_est[key] = emiss_counts[key] / (ngram_counts[(key[0],)] or num_total_words)

    return qml_est, eml_est

def get_discounted_estimates(ngram_counts, tag_set, beta=0.5):
    """
    Get discounted ML estimates for transition and emission
    probabilities.
    """

    qd_est = defaultdict(lambda: 0)
    alpha = defaultdict(lambda: 0)

    # get discounted estimates
    for ngram in ngram_counts.keys():
        if not ngram_counts[ngram] or len(ngram) < 2: continue

        qd_est[ngram] = (ngram_counts[ngram] - beta) / (ngram_counts[ngram[:-1]] or ngram_counts[('STOP',)])

        #if ngram[:-1] == ('DT','DT'):
        #    print qd_est[ngram], ngram, ngram[-1], ngram_counts[ngram], ngram_counts[ngram[:-1]]

        alpha[ngram[:-1]] += qd_est[ngram]

    # create missing mass
    for ngram in alpha.keys():
        alpha[ngram] = 1 - alpha[ngram]

    # divide missing mass in proportion to ngram estimates
    for i in xrange(2,2+order):

        # calculate denominator (sum of either ML or qd estimates)
        if i == 2:
            denom_counts = \
            {(v,):sum([ngram_counts[w,] if ngram_counts[v,w] == 0\
                       else 0 for w in tag_set]) for v in tag_set}
        else:
            tag_set_list = [tag_set - {'STOP',}] * (i-2) + [tag_set]
            denom_counts = \
            {ngram:sum([qd_est[ngram] if ngram_counts[ngram+tuple(w)] == 0\
                        else 0 for w in tag_set]) for ngram in product(*tag_set_list)}

        # calculate values for unseen ngrams
        tag_set_list = [tag_set - {'STOP',}] * (i-1) + [tag_set]
        for ngram in product(*tag_set_list):

            first = ngram[:-1]
            last = ngram[1:]

            if not ngram_counts[ngram] and i == 2:
                qd_est[ngram] = alpha[first] * ngram_counts[last] / denom_counts[first]

            elif not ngram_counts[ngram]:
                qd_est[ngram] = alpha[first] * qd_est[last] / denom_counts[first]

    return qd_est

def map_to_pseudo_word(word, k):
    """
    Map low-frequency words at position k to appropriate
    pseudo-words.
    """
    
    if k == 1:                                          # first word in sentence
        pseudo = '__firstWord__'
    elif re.match(r'[A-Z]+$', word):                    # organization
        pseudo = '__allCaps__'
    elif re.match(r'[A-Z]\.$', word):                   # person name initial
        pseudo = '__capPeriod__'
    elif re.match(r'[A-Z]\w*$', word):                  # capitalized word
        pseudo = '__initCap__'
    elif re.match(r'[a-z]\w*$', word):                  # uncapitalized word
        pseudo = '__lowercase__'
    elif re.match(r'\$[0-9][0-9,.]*$', word):           # monetary amount (dollars)
        pseudo = '__currency__'
    elif re.match(r'[0-9]+[0-9-/.,A-Za-z]*$', word):    # numeric value
        pseudo = '__number__'
    else:                                               # other
        pseudo = '__other__'
    
    return pseudo

def run_viterbi(word_seq, q_params, e_params, tag_set, use_pseudo=False, order=2):
    """
    Implementation of the Viterbi algorithm with backpointers. Returns
    the tags that maximize the probability of a given sentence occuring,
    based on an n-gram hidden markov model.

    Args:
        q_params (defaultdict): Transition probabilities. q(s|u,v) -> q_params(u,v,s)
        e_params (defaultdict): Emission probabilities. e(x|s) -> e_params(s,x)
        word_seq (list): The sentence to be tagged.
        tag_set (set): Set of potential tags for a word.
        order (int): The order of the Markov sequence. Defaults to 2.
    """
    
    # base case for pi parameters, initialize backpointers
    pi_params = {(0,('*',) * n):1 for n in xrange(1,order+1)}
    bp_params = {}
    
    # probability words are seen with tags
    sent_prob = 0

    for k,word in enumerate(word_seq, start=1):

        # create list of tag sets for words at position k-order+1 to k
        tag_set_list = [tag_set if i+1 > 0 else {'*'} for i in xrange(k-order,k)]

        # create set of all possible tag sequences that end at position k
        tag_seq_list = {tag_seq for tag_seq in product(*tag_set_list)}
        
        # map unseen words to pseudo-words
        if use_pseudo and e_params['_',word] == 0:
            word = map_to_pseudo_word(word, k)

        # iterate through tag sequences that end at position k
        for tag_seq in tag_seq_list:

            pi_params[k, tag_seq] = 0   # initialize pi parameter

            # loop through set of tags in leftmost position
            for tag in k-order > 0 and tag_set or {'*'}:

                pi_key = k-1, (tag,) + tag_seq[:-1]     # looks like k-1,(w,u)
                q_key = (tag,) + tag_seq                # looks like w,u,v -- transition param v|w,u
                e_key = tag_seq[-1], word               # looks like v,x -- emission param x|v

                temp = pi_params[pi_key] * q_params[q_key] * e_params[e_key]
                
                # get largest pi parameter and remember arguments
                if temp > pi_params[k, tag_seq]:
                    pi_params[k, tag_seq] = temp
                    bp_params[k, tag_seq] = tag
                    
            if k == len(word_seq):

                temp_sent_prob = pi_params[k, tag_seq] * q_params[tag_seq + ('STOP',)]

                if temp_sent_prob > sent_prob:
                    sent_prob = temp_sent_prob
                    final_tag_seq = tag_seq
    
    # were we able to tag the sentence?
    if not sent_prob:
        return ['*'] * k
    
    # retrieve tags for each word in the sentence
    final_tag_seq = list(final_tag_seq)
    for k in xrange(k-order, 0, -1):
        final_tag_seq.insert(0, bp_params[k+order, tuple(final_tag_seq[:order])])

    return final_tag_seq




def run_baseline(word_seq, e_counts, tag_set):
    return [max({(tag,word):emiss_counts[tag,word] \
                 for tag in tag_set}.iteritems(), key=lambda x:x[1])[0][0] for word in word_seq]

def score_tagger(infile, tagger, *params):
    """
    description goes here
    Precision: # of correctly tagged words / # of tagged words
    Recall: # of tagged words / # of words
    """
    
    # store counts of tagged words
    num_correct_words = num_tagged_words = num_words = 0

    with open(infile) as corpus:
        word_seq = []
        answer = []
        
        for line in corpus:
            
            if line == '\n':

                # if there is no data, just skip
                if word_seq == []: continue

                # tagger will return either a list of tags (ex. ['DT','NN','VB'])
                # or ['*','*',...'*'] if sentence could not be tagged
                result = tagger(word_seq, *params)
                
                # get number of correctly tagged words
                num_correct_words += sum([result[i] == answer[i] for i in xrange(len(answer))])
                num_tagged_words += len(result) if result[0] != '*' else 0
                num_words += len(result)

                # reset sentence
                word_seq = []
                answer = []

                continue

            # get word and tag, add to list
            word, tag, chunk = line.rstrip().split(' ')
            word_seq.append(word)
            answer.append(tag)

    # calculate precision and recall
    precision_words = num_correct_words/num_tagged_words
    recall_words = num_tagged_words/num_words
    fscore_words = 2*precision_words*recall_words/(precision_words+recall_words)
    
    return precision_words, recall_words, fscore_words
