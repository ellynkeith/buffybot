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
            line = line.lower()
            line = line.split(" ")
            line.insert(0,"<s>")
            line.append("</s>")
            prepScript.append(line)

        listLines.append(prepScript) 

    ## collapse list of lists into single list    
    all_lines = [lines for script in listLines for lines in script]  

    buffBGs = []
    for line in all_lines:
        bgs = list(ngrams(line, 2))

        for b in bgs:
            buffBGs.append(b)

    seed = "<s>"
    while seed != "</s>":
        word = chooseWord(seed, buffBGs)
        print(word, end=" ")
        seed = word

main()






