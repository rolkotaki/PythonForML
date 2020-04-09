import re
from bs4 import BeautifulSoup
import unicodedata
import sys
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk import pos_tag
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# nltk.download()


# Cleaning text
from sklearn.preprocessing import MultiLabelBinarizer

text_data = ["   Interrobang. By Aishwarya Henriette     ",
             "Parking And Going. By Karl Gautier",
             "    Today Is The night. By Jarek Prakash   "]

# Strip whitespaces
strip_whitespace = [string.strip() for string in text_data]
# removing periods
remove_periods = [string.replace(".", "") for string in strip_whitespace]
# custom transformation to capitalize
def capitalizer(string: str) -> str:
    return string.upper()

capitalized = [capitalizer(string) for string in remove_periods]
print(capitalized)

# transformation using regular expressions
def replace_letters_with_X(string: str) -> str:
    return re.sub(r"[a-zA-Z]", "X", string)

# Apply function
print([replace_letters_with_X(string) for string in remove_periods])


# Parsing and Cleaning HTML

html = """
       <div class='full_name'><span style='font-weight:bold'>
       Masego</span> Azra</div>"
       """
# Parse html
soup = BeautifulSoup(html, "lxml")

# Find the div with the class "full_name"
print(soup.find("div", { "class" : "full_name" }).text)


# Removing Punctuation

text_data = ['Hi!!!! I. Love. This. Song....',
             '10000% Agree!!!! #LoveIT',
             'Right?!?!']
# Create a dictionary of punctuation characters
punctuation = dict.fromkeys(i for i in range(sys.maxunicode)
                            if unicodedata.category(chr(i)).startswith('P'))
# For each string, remove any punctuation characters
print([string.translate(punctuation) for string in text_data])


# Tokenizing Text (breaking it up, splitting)

# Tokenize words
string = "The science of today is the technology of tomorrow"
print(word_tokenize(string))
# Tokenize sentences
string = "The science of today is the technology of tomorrow. Tomorrow is today."
print(sent_tokenize(string))


# Removing Stop Words

# NLTK’s stopwords assumes the tokenized words are all lowercase !!!!!!!!!!!!!!!!!
tokenized_words = ['i',
                   'am',
                   'going',
                   'to',
                   'go',
                   'to',
                   'the',
                   'store',
                   'and',
                   'park']
# Load stop words
stop_words = stopwords.words('english')
# Remove stop words
print([word for word in tokenized_words if word not in stop_words])


# Stemming Words

# NLTK’s PorterStemmer implements the widely used Porter stemming algorithm to remove or replace common suffixes
# to produce the word stem

tokenized_words = ['i', 'am', 'humbled', 'by', 'this', 'traditional', 'meeting']
# Create stemmer
porter = PorterStemmer()
# Apply stemmer
print([porter.stem(word) for word in tokenized_words])


# Tagging Parts of Speech

text_data = "Chris loved outdoor running"
# Use pre-trained part of speech tagger
text_tagged = pos_tag(word_tokenize(text_data))  # will output a list of tuples
# Show parts of speech
print(text_tagged)
# Once the text has been tagged, we can use the tags to find certain parts of speech
print([word for word, tag in text_tagged if tag in ['NN', 'NNS', 'NNP', 'NNPS'] ])

# An example of working with tweets:
tweets = ["I am eating a burrito for breakfast",
          "Political science is an amazing field",
          "San Francisco is an awesome city"]
# Create list
tagged_tweets = []
# Tag each word and each tweet
for tweet in tweets:
    tweet_tag = pos_tag(word_tokenize(tweet))
    tagged_tweets.append([tag for word, tag in tweet_tag])

# Use one-hot encoding to convert the tags into features
one_hot_multi = MultiLabelBinarizer()
print(one_hot_multi.fit_transform(tagged_tweets))

# Show feature names
print(one_hot_multi.classes_)


# Encoding Text as a Bag of Words

# You have text data and want to create a set of features indicating the number of times
# an observation’s text contains a particular word.

text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])
# Create the bag of words feature matrix
count = CountVectorizer()
bag_of_words = count.fit_transform(text_data)
# CountVectorizer returns a sparse matrix, which is very good !!!
print(bag_of_words)
print(bag_of_words.toarray())  # containing 0s as well for better view
# Show feature names
print(count.get_feature_names())
print(count.vocabulary_)  # to view the word associated with each feature

# Other example

# Create feature matrix with arguments
count_2gram = CountVectorizer(ngram_range=(1, 2),     # creating features with a combination of 2 words
                              stop_words="english",   # removing stop words
                              vocabulary=['brazil'])  # to restrict the words we want to consider; ie: list of countries
bag = count_2gram.fit_transform(text_data)
# View feature matrix
print(bag.toarray())
print(count_2gram.vocabulary_)


# Weighting Word Importance

# You want a bag of words, but with words weighted by their importance to an observation.

text_data = np.array(['I love Brazil. Brazil!',
                      'Sweden is best',
                      'Germany beats both'])
# Create the tf-idf feature matrix
tfidf = TfidfVectorizer()
feature_matrix = tfidf.fit_transform(text_data)
# Show tf-idf feature matrix
feature_matrix  # this is a sparse matrix
print(feature_matrix.toarray())
print(tfidf.vocabulary_)
