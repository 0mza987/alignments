
# https://towardsdatascience.com/pairwise-sequence-alignment-using-biopython-d1a9d0ba861f

from Bio.pairwise2 import format_alignment
from Bio import pairwise2

LIST_pair = list()


# Define two sequences to be aligned
X = "My name is Tang Lo I was born in a small town in thejung"
Y = "My nome is Tang Li . I was born in a small torn in Zhejing"
LIST_pair.append((X, Y))

X = "Province I'm 12 years old I was born on loth June , He My frost"
Y = "Prorince I'm 1 years old I was born on 1oth. June , 110. My frist"
LIST_pair.append((X, Y))

X = "teacher was Guo va My first school was Chenguang Primary"
Y = "teacher was Guo Xia . My | first school was Chenguang Primary"
LIST_pair.append((X, Y))

X = "Schol I'm kind and friendly My belly is playing football and dancing"
Y = "School . I'm kind and friendly . My holby is playing football and doncing ."
LIST_pair.append((X, Y))

X = "I'm want to be a dancer this is me How about you"
Y = "I'm want to be a dancer. This is me. How |out you?"
LIST_pair.append((X, Y))


'''
My name is Tang Lo I was born in a small town in thejung
Province I'm 12 years old I was born on loth June , He My frost
teacher was Guo va My first school was Chenguang Primary
Schol I'm kind and friendly My belly is playing football and dancing
I'm want to be a dancer this is me How about you

*** My nome is Tang Li . I was born in a small torn in Zhejing
*** Prorince I'm 1 years old I was born on 1oth. June , 110. My frist
*** teacher was Guo Xia . My | first school was Chenguang Primary
*** School . I'm kind and friendly . My holby is playing football and doncing .
*** I'm want to be a dancer. This is me. How |out you?

'''

for X, Y in LIST_pair:

    # Get a list of the global alignments between the two sequences ACGGGT and ACG satisfying the given scoring
    # A match score is the score of identical chars, else mismatch score.
    # Same open and extend gap penalties for both sequences.
    alignments = pairwise2.align.globalms(X, Y, 2, -1, -2, -0.1)
    # alignments = pairwise2.align.localxx(X, Y)

    # Use format_alignment method to format the alignments in the list
    a = alignments[-1]
    # for a in alignments:
    print format_alignment(*a)
