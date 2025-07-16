import random
from nltk.util import ngrams
from nltk import word_tokenize
import os
import glob

## Function to return words from bigrams
def chooseWord(seed, bgrams):
    words = [pair[1] for pair in bgrams if pair[0] == seed]
    random.shuffle(words)

    return words[0]

def buffBigrams(list_text):
    all_BGs = []
    for line in list_text:
        bgs = list(ngrams(line, 2))

        ## put all bigrams in single list
        for b in bgs:
            all_BGs.append(b)
    return all_BGs

def chooseWordTrigrams(firstWord, secWord, TGs):
    tri = [words[2] for words in TGs if words[0] == firstWord and words[1] == secWord]
    random.shuffle(tri)

    return tri[0]

def main():
    path = "BUFFY/"
    listLines = []

    ## for each text file, read as string, split on newline char
    for filename in glob.glob(os.path.join(path, "*.txt")):
        getfile = open(filename, "r")
        script = getfile.read().split("\n")
        getfile.close()

    ## for each line in script, convert to lowercase, split into list of strings
    ## add start/end sentence tags 
        prepScript = []
        for line in script:
 #           line = line.lower()
            line = line.split(" ")
            line.insert(0,"<s>")
            line.append("</s>")
            prepScript.append(line)

        listLines.append(prepScript) 

    ## collapse list of lists into single list    
    all_lines = [lines for script in listLines for lines in script]  
    print(all_lines[100])

    ## create bigrams from processed text
    buffBGs = buffBigrams(all_lines)

    TGlist = []
    for line in all_lines:
        tgs = list(ngrams(line,3))

        for t in tgs:
            TGlist.append(t)

    firstWord = chooseWord("<s>", TGlist)
    seed1 = "<s>"
    seed2 = firstWord
    sent = []
    #print(firstWord, end = " ")
    # sent.append(firstWord)
    # while seed1 != "</s>":
    #     word = chooseWord(seed1, buffBGs)
    #     sent.append(word)
    #     print(word, end=" ")
    #     seed = word

    while seed2 != "</s>":
        word = chooseWordTrigrams(seed1, seed2, TGlist)
        sent.append(word)
        print(word, end = " ")
        seed1, seed2 = seed2, word
 

main()






