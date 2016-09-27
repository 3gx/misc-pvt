#!/usr/bin/env python

def build_dictionary(text):
    data = text.splitlines()

    fn_dict = {}
    fi_dict = {}

    for line in data:
        hasfn = False
        hasfi = False
        if len(line) >= 3:
            if line[0:3] == 'fn=':
                hasfn = True
            elif line[0:3] == 'fl=' or line[0:3] == 'fi=':
                hasfi = True

        if len(line) >= 4:
            if line[0:4] == 'cfn=':
                hasfn = True
            elif line[0:4] == 'cfl=' or line[0:4] == 'cfi=':
                hasfi = True

        if hasfn:
            words = line.split();
            if len(words) >= 2:
                w = words[0].split("=")
                fn_dict[w[1]] = " ".join(words[1:])
        
        if hasfi:
            words = line.split();
            if len(words) >= 2:
                w = words[0].split("=")
                fi_dict[w[1]] = " ".join(words[1:])


    return [fn_dict, fi_dict]

def transform_text(text, dictionary):
    text1 = []
    data = text.splitlines()

    fn_dict = dictionary[0]
    fi_dict = dictionary[1]

    for line in data:
        hasfn = False
        hasfi = False
        hascfn = False
        hascfi = False
        if len(line) >= 3:
            if line[0:3] == 'fn=':
                hasfn = True
            elif line[0:3] == 'fl=' or line[0:3] == 'fi=':
                hasfi = True

        if len(line) >= 4:
            if line[0:4] == 'cfn=':
                hascfn = True
            elif line[0:4] == 'cfl=' or line[0:4] == 'cfi=':
                hascfi = True

        if hasfn:
            words = line.split();
            if len(words) >= 1:
                w = words[0].split("=")
                w[1] = fn_dict[w[1]]
                words[0] = "func= "+w[1]
            line = words[0]
        elif hascfn:
            words = line.split();
            if len(words) >= 1:
                w = words[0].split("=")
                w[1] = fn_dict[w[1]]
                words[0] = "cfunc= "+w[1]
            line = words[0]
        elif hasfi:
            words = line.split();
            if len(words) >= 1:
                w = words[0].split("=")
                w[1] = fi_dict[w[1]]
                words[0] = "func_i= "+w[1]
            line = words[0]
        elif hascfi:
            words = line.split();
            if len(words) >= 1:
                w = words[0].split("=")
                w[1] = fi_dict[w[1]]
                words[0] = "cfunc_i= "+w[1]
            #line = " ".join(words)
            line = words[0]

        text1 += [line]

    return "\n".join(text1)


import sys

text_in = sys.stdin.read()
dictionary = build_dictionary(text_in)
text_out = transform_text(text_in, dictionary)
print text_out


