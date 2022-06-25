from nltk.corpus import names
import nltk
nltk.download('names')
male1 = nltk.Text(nltk.corpus.names.words('male.txt'))
male1
len(male1)
female1 = nltk.Text(nltk.corpus.names.words('female.txt'))
female1
len(female1)
labeled_names = ([(name, 'male') for name in names.words('male.txt')]+ [(name, 'female') for name in names.words('female.txt')])
labeled_names
import random
random.shuffle(labeled_names)
#Step 2 Create the feature extractor function and the feature set
#Names ending in a, e and i are likely to be female, while names ending in k, o, r, s and t are likely to be male. Let's build a classifier to model these differences more precisely.
def feature_extractor(name):
    return {'last_letter': name[-1]}
featureset = [(feature_extractor(name), gender) for (name, gender) in labeled_names]
featureset
len(featureset)
#Step 3 Feature set is divided into training and testing dataset
train_set = featureset[:500]
train_set
len(train_set)
test_set = featureset[500:]
test_set
len(test_set)
#Step 4 Build the classsifier
import nltk
classifier = nltk.NaiveBayesClassifier.train(train_set)
#Step 5 
#Evaluate the Classifier
print(nltk.classify.accuracy(classifier, test_set))
#Test the Classifier
classifier.classify(feature_extractor('Neo'))
classifier.classify(feature_extractor('Trinity'))
#Finally, we can examine the classifier to determine which features it found most effective for distinguishing the names' genders:
classifier.show_most_informative_features(5)

#MOVIE REVIEW SENTIMENT ANALYSIS
#Step 01: Create a python file and import the following packages.
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
#Step 02: Define a function to extract features.
def extract_features(word_list):
    return dict([(word, True) for word in word_list])
#Step 03: To get the training data, use the following movie reviews from NLTK.
if __name__=='__main__':
   # Load positive and negative reviews  
   positive_fileids = movie_reviews.fileids('pos')
   negative_fileids = movie_reviews.fileids('neg')

#Step 04: Now we will separate the positive and negative reviews.
features_positive = [(extract_features(movie_reviews.words(fileids=[f])),'Positive') for f in positive_fileids]
features_negative = [(extract_features(movie_reviews.words(fileids=[f])),'Negative') for f in negative_fileids]
#Step 05: Since we need 2 datasets for this process, divide the data into training and testing datasets.
# Split the data into train and test (80/20)
threshold_factor = 0.8
threshold_positive = int(threshold_factor * len(features_positive))
threshold_negative = int(threshold_factor * len(features_negative))
#Step 06: Extract the features.
features_train = features_positive[:threshold_positive]+features_negative[:threshold_negative]
features_test = features_positive[threshold_positive:]+features_negative[threshold_negative:]
print("Number of training datapoints: ", len(features_train))
print("Number of test datapoints: ", len(features_test)) 
#Step 07: Use the Navie Bayes classifier. Define the object and train it.
classifier = NaiveBayesClassifier.train(features_train)
print("Accuracy of the classifier: ", nltk.classify.util.accuracy(classifier, features_test))

#Step 08: To find out the most informative words inside the classifier which decides a review is positive or negative, print the following.
print("Top ten most informative words: ")
for item in classifier.most_informative_features()[:10]:
    print(item[0])

#Step 9:Create some random movie reviews of your own.
    #Sample input reviews
input_reviews = [
    "Started off as the greatest series of all time, but had the worst ending of all time.",
    "Exquisite. 'Big Little Lies' takes us to an incredible journey with its emotional and intriguing storyline.",
    "I love Brooklyn 99 so much. It has the best crew ever!!",
    "The Big Bang Theory and to me it's one of the best written sitcoms currently on network TV.",
    "'Friends' is simply the best series ever aired. The acting is amazing.",
    "SUITS is smart, sassy, clever, sophisticated, timely and immensely entertaining!",
    "Cumberbatch is a fantastic choice for Sherlock Holmes-he is physically right (he fits the traditional reading of the character) and he is a damn good actor",
    "What sounds like a typical agent hunting serial killer, surprises with great characters, surprising turning points and amazing cast."
    "This is one of the most magical things I have ever had the fortune of viewing.",
    "I don't recommend watching this at all!"
]

#Step 10: Now, run the classifier on those sentences and obtain the predictions.
print("Predictions: ")

for review in input_reviews:
    print("\nReview:", review)
    probdist = classifier.prob_classify(extract_features(review.split()))
    pred_sentiment = probdist.max()


#Step 11:It’s done. Now you can print the output.

print("Predictions: ")
for review in input_reviews:
    print("\nReview:", review)
    probdist = classifier.prob_classify(extract_features(review.split()))
    pred_sentiment = probdist.max()
    print("Predicted sentiment: ", pred_sentiment)
    print("Probability: ", round(probdist.prob(pred_sentiment), 2))
    
    
    
#Bigrams
from nltk import bigrams,trigrams
from nltk.tokenize import word_tokenize
text = " Dostoevsky,was the son of the doctor"
ngrams=trigrams(word_tokenize(text))
list(ngrams)

#https://grammar.yourdictionary.com/for-students-and-parents/words-with-multiple-meanings.html
#https://www.analyticsvidhya.com/blog/2017/06/word-embeddings-count-word2veec/

    
    
