#!/usr/bin/env python

def build_dictionary(text):
    data = text.splitlines()

    dictionary = {}

    for line in data:
        hasfn = False
        if len(line) >= 3:
            if line[0:3] == 'fn=':
                hasfn = True
        if len(line) >= 4:
            if line[0:4] == 'cfn=':
                hasfn = True

        if hasfn:
            words = line.split();
            if len(words) == 2:
                w = words[0].split("=")
                dictionary[w[1]] = words[1]

    return dictionary

def transform_text(text, dictionary):
    text1 = ""

    return text1


import sys

fin = sys.argv[1]
fout = sys.argv[2]

print "input file: "+fin
print "output file: "+fout

f = open(fin,"r")
text = f.read();
dictionary = build_dictionary(text)
text1 = transform_text(text, dictionary)


