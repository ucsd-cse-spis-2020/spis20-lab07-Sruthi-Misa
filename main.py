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


#classify
import nltk 
nltk.download("stopwords")
nltk.download('punkt')
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.classify.util import accuracy

# "Stop words" that you might want to use in your project/an extension
stop_words = set(stopwords.words('english'))

def format_sentence(sent):
    ''' format the text setence as a bag of words for use in nltk'''
    tokens = nltk.word_tokenize(sent)
    return({word: True for word in tokens})

def getReviews(data, rating):
    ''' Return the reviews from the rows in the data set with the
        given rating '''
    rows = data['Rating']==rating
    return list(data.loc[rows, 'Review'])

def splitTrainTest(data, trainProp):
    ''' input: A list of data, trainProp is a number between 0 and 1
              specifying the proportion of data in the training set.
        output: A tuple of two lists, (training, testing)
    '''
    # TODO: You will write this function, and change the return value
    num = int(len(data)*trainProp)
    list1 = data[:num]
    list2 = data[num:]
    
    return (list1, list2)

#data = ['A', 'B', 'C', 'D']
#print(splitTrainTest(data, 0.5))

def formatForClassifier(dataList, label):
    ''' input: A list of documents represented as text strings
               The label of the text strings.
        output: a list with one element for each doc in dataList,
                where each entry is a list of two elements:
                [format_sentence(doc), label]
    '''
    # TODO: Write this function, change the return value
    classifier_list = []
    for x in range(len(dataList)):
        new_dict = format_sentence(dataList[x])
        classifier_list.append([new_dict, label])
    return classifier_list
#print(formatForClassifier(["A good one", "The best!"], "pos"))

def classifyReviews():
    ''' Perform sentiment classification on movie reviews ''' 
    # Read the data from the file
    data = pd.read_csv("data/movieReviews.csv")

    # get the text of the positive and negative reviews only.
    # positive and negative will be lists of strings
    # For now we use only very positive and very negative reviews.
    positive = getReviews(data, 4)
    negative = getReviews(data, 0)

    # Split each data set into training and testing sets.
    # You have to write the function splitTrainTest
    (posTrainText, posTestText) = splitTrainTest(positive, 0.8)
    (negTrainText, negTestText) = splitTrainTest(negative, 0.8)

    # Format the data to be passed to the classifier.
    # You have to write the formatForClassifer function
    posTrain = formatForClassifier(posTrainText, 'pos')
    negTrain = formatForClassifier(negTrainText, 'neg')

    # Create the training set by appending the pos and neg training examples
    training = posTrain + negTrain

    # Format the testing data for use with the classifier
    posTest = formatForClassifier(posTestText, 'pos')
    negTest = formatForClassifier(negTestText, 'neg')
    # Create the test set
    test = posTest + negTest


    # Train a Naive Bayes Classifier
    # Uncomment the next line once the code above is working
    classifier = NaiveBayesClassifier.train(training)

    # Uncomment the next two lines once everything above is working
    print("Accuracy of the classifier is: " + str(accuracy(classifier, test)))
    classifier.show_most_informative_features()

    # TODO: Calculate and print the accuracy on the positive and negative
    # documents separately
    # You will want to use the function classifier.classify, which takes
    # a document formatted for the classifier and returns the classification of that document ("pos" or "neg"). 
    #For example:
    # will (hopefully!) return "pos"

    # TODO: Print the misclassified examples


if __name__ == "__main__":
    classifyReviews()