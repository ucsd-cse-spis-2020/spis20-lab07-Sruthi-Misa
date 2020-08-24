import pprint
import random

def train(s):
    words = s.replace('\n', '.').split()
    word_dict = {}
    for i in range(len(words)):
        current_word = words[i]
        if i == 0:
            previous_word = None
        else:
            previous_word = words[i-1]
        if previous_word not in word_dict:
            word_dict[previous_word] = []
        word_dict[previous_word].append(current_word)
    pp = pprint.PrettyPrinter(indent=1)
    #pp.pprint(word_dict)
    return word_dict

def generate(model, firstWord, numWords):
    nextWord = None
    for x in range(numWords):
        nextWord = random.choice(model[firstWord])
        print(firstWord, end=" ")
        firstWord = nextWord

peach_fuzz = train('''
I was in the corner Drinking from the punch Yeah, you were in the kitchen Cuttin' up a rug No need to complicate it I had fallen in love With you, so underrated Something fillin' up my lungs Every color of your love, I've seen enough, I want another Every color of your love, I've seen enough, I want another Hey little mamma when you talk back I see your eyes light up and I love that I'm just a peach fuzz boy, I'm so alone I don't wanna miss you honey, come home Knock knock, you're coming over Couple times a week Just hanging on my shoulder Come on, shaking like a leaf Every color of your love, I've seen enough, I want another Every color of your love, I've seen enough, I want another''')

generate(peach_fuzz, "peach", 20)