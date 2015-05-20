#! /usr/bin/python

import re
import sys
from collections import defaultdict

"""
Count word frequencies in document and replace low-frequency words
with pseudo-words (allCaps, initCap, firstWord, etc). Write lines
to stdout with replaced words.
"""

def get_counts(corpus):
    """
    Iterate through each line of input file, counting words and
    storing frequencies in dictionary.
    """

    # create dictionary to store word counts
    words = defaultdict(lambda: 0)

    # iterate through lines in file
    for line in corpus:

        # skip blank lines
        if line == '\n': continue

        # increment count in dictionary
        word = line.split(' ', 1)[0]
        words[word] += 1

    # put cursor back at start of file
    corpus.seek(0)

    return words

def replace_words(corpus, words, output, threshold=5):
    """
    Replace low-frequency words with appropriate pseudo-words. Threshold
    argument is the frequency below which words should be replaced.
    """

    new_sent = True

    # iterate through lines in file
    for line in corpus:

        # if line is blank, set "new sentence" flag to True
        if line == '\n':
            new_sent = True
            output.write('\n')
            continue

        word, tags = line.split(' ', 1)

        # if word occurs below threshold, map to pseudo-word
        if words[word] < threshold:

            if new_sent:                                        # first word in sentence
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

            new_line = pseudo + ' ' + tags

        else:
            new_line = line

        output.write(new_line)

        if new_sent: new_sent = False

    # put cursor back at start of file
    corpus.seek(0)

def usage():
    print """
    python replace_words.py input_file [threshold] > output_file
        Read input file and replace low-frequency words with
        pseudo-words. Threshold argument is the frequency below
        which words should be replaced (default is five).
    """

if __name__ == "__main__":

    # user has provided a frequency threshold
    if len(sys.argv) == 3:

        try:
            threshold = int(sys.argv[2])
        except ValueError:
            usage()
            sys.exit(2)

    # user has entered either zero or more than two arguments
    elif len(sys.argv) < 2 or len(sys.argv) > 3:
        usage()
        sys.exit(2)

    # user has not provided a frequency threshold, use default
    else:
        threshold = 5

    # check to make sure we can open file
    try:
        corpus = open(sys.argv[1], 'r')
    except IOError:
        usage()
        sys.exit('ERROR: Could not read input file "' + sys.argv[1] + '"')

    words = get_counts(corpus)
    replace_words(corpus, words, sys.stdout, threshold)
    corpus.close()