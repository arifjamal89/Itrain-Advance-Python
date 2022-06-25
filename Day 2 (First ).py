
text = """ Dostoevsky,was the son of the doctor…..."""
#List The vocabulary
vocab = sorted(set(text.lower().split()))
vocab
#Count the vocabulary
len(vocab)
#Count the occuranece
text.count('the')
#Filter Words
text.split(".")
text.split()

stop_word=[".",",","a","the"]
a=[word for word in text if word not in stop_word]
a


#Example

from urllib import request
url = "http://www.gutenberg.org/files/2554/2554-0.txt"
response = request.urlopen(url)
#Encode the data 
encoding = response.info().get_param('charset', 'utf8')
text1 = response.read().decode(encoding)
text1
# Count total no of words
vocab1 = sorted(set(text1.lower().split()))
len(vocab1)
longword=[word for word in vocab1 if len(word)>5]
longword

# Tokenize special words

from nltk.tokenize import sent_tokenize, word_tokenize
print(len(sent_tokenize(text1)))
print(len(word_tokenize(text1)))


#Tokenzing
from nltk.tokenize import sent_tokenize, word_tokenize

EXAMPLE_TEXT = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."

print(sent_tokenize(EXAMPLE_TEXT))
print(word_tokenize(EXAMPLE_TEXT))

#Or the word tokenize can be usedlike this
for i in word_tokenize(EXAMPLE_TEXT):
    print(i)


#nltk import
#pip install nltk
import nltk
nltk.download('all')


#STOP WORDS
from nltk.corpus import stopwords

#list the default stopwords
stop_words=set(stopwords.words('english'))

#Example
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "This is a sample sentence, showing off the stop words filtration."

stop_words = set(stopwords.words('english'))
print(stop_words)
word_tokens = word_tokenize(example_sent)
print(word_tokens)
#Create a filtered sentence
filtered_sentence=[]
for w in word_tokens:
    if w not in stop_words:
        filtered_sentence.append(w)

# or the above code can be replaced as
#   filtered_sentence = [w for w in word_tokens if not w in stop_words]
print(filtered_sentence)


# Remove Punctuation from text\
from string import punctuation
def strip_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)
text = "This is a sample sentence, showing off the stop words filtration."
print (strip_punctuation(text))
   
# Remove numbers from text
text = "There was 200 people standing right next to me at 2pm."
output = ''.join(c for c in text if not c.isdigit())
print(output)


#Remove Html tags in a text
import re
text = """<head><body>hello world!</body></head>"""
cleaned_text = re.sub('<[^<]+?>','', text)
print (cleaned_text)

#Regular Expression for detecting Word Patterns
import re
import nltk
wordlist = [w for w in nltk.corpus.words.words('en') if w.islower()]#List the words in the English Dictionary words
a=[w for w in wordlist if re.search('ed$', w)]#List the words in the wordlist ending with ed
len(a)
#The . wildcard symbol matches any single character. Suppose we have room in a crossword puzzle for an 8-letter word with j as its third letter and t as its sixth letter. In place of each blank cell we use a period:
[w for w in wordlist if re.search('^..j..t..$', w)]
[w for w in wordlist if re.match('^[a-z]+$',w)]


#Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()
example_words = ["python","pythoner","pythoning","pythoned","pythonly"]
for w in example_words:
    print(ps.stem(w))
    
#Example stemming
new_text = "It is important to by very pythonly while you are pythoning with python. All pythoners have pythoned poorly at least once."
words=word_tokenize(new_text)
for w in words:
    print(ps.stem(w))


#Lemmatize

import nltk
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

sentence = "He was running and eating at same time. He has bad habit of swimming after playing long hours in the Sun."
punctuations="?:!.,;"
sentence_words = nltk.word_tokenize(sentence)
for word in sentence_words:
    if word in punctuations:
        sentence_words.remove(word)

sentence_words
print("{0:20}{1:20}".format("Word","Lemma"))
for word in sentence_words:
    print ("{0:20}{1:20}".format(word,wordnet_lemmatizer.lemmatize(word,pos="v")))



#POS
nltk.pos_tag(['cat','cats'])
nltk.pos_tag(['take','took','taking','taken'])
nltk.pos_tag(['delicious'])
nltk.pos_tag(['slowly'])
text = word_tokenize("And now for something completely different")
nltk.pos_tag(text)




#EXAMPLE
from urllib import request
url = "https://en.wikipedia.org/wiki/George_Washington"
response = request.urlopen(url)
#Encode the data 
encoding = response.info().get_param('charset', 'utf8')
text1 = response.read().decode(encoding)
text1
from nltk.tokenize import sent_tokenize, word_tokenize
print(len(sent_tokenize(text1)))
print(len(word_tokenize(text1)))
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
stop_words = set(stopwords.words('english'))
text2=strip_punctuation(text1)
wordnet_lemmatizer = WordNetLemmatizer()
word_tokens = nltk.word_tokenize(text2)
word_tokens1=''.join(c for c in word_tokens if not c.isdigit())
print(word_tokens1)
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in word_tokens]
print (lemmatized_word)
nltk.pos_tag(text2)




#brown
#In the rest of this chapter we will explore various ways to automatically add part-of-speech tags to text. We will see that the tag of a word depends on the word and its context within a sentence. For this reason, we will be working with data at the level of (tagged) sentences rather than words. We'll begin by loading the data we will be using.
from nltk.corpus import brown
brown_sents = brown.sents(categories='news')
brown_sents
brown_tagged_sents = brown.tagged_sents(categories='news')
brown_tagged_sents



#MODULE 3


from nltk.corpus import gutenberg
import nltk
nltk.download('gutenberg')
import nltk
nltk.download('punkt')
fileid = 'austen-emma.txt'
text = gutenberg.raw(fileid)
#Fileid : 
gutenberg.fileids()
#Text : 
gutenberg.raw(fileid)
#Words : 
gutenberg.words(fileid)
#Sentence : 
gutenberg.sents(fileid)
from nltk.tokenize import sent_tokenize
tok = sent_tokenize(text)

for x in range(5):
    print(tok[x])
    
from nltk.corpus import brown
brown.categories()
from nltk.corpus import brown
brown.categories()
text = brown.raw(categories='news')


from nltk.corpus import reuters
reuters.fileids()
reuters.categories()
fileid='test/16399'
text = reuters.raw(fileid)
text1=reuters.raw(categories='zinc')



from nltk.corpus import movie_reviews
movie_reviews.fileids()
movie_reviews.categories()
fileid=''
text = movie_reviews.raw(fileid)
text1=movie_reviews.raw(categories='')


#Frequency distribution by creating our own corpus

from nltk.corpus import PlaintextCorpusReader
fileid = 'C:/Users/arun/Desktop/ITRAIN/itrain python/Advanced/codes/DAY 2/gaming.txt'
my_corpus = PlaintextCorpusReader(fileid, '.*')
text = my_corpus.raw(fileid)
text
my_corpus.fileids()
my_corpus.raw(fileid)
my_corpus.words(fileid)
my_corpus.sents(fileid)
distr = nltk.FreqDist(text)
print(distr.most_common(5))



#Reuters
from nltk.corpus import reuters
fileid='training/9865'
text=reuters.raw(fileid)
text
