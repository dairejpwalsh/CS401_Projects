import os
import re
import urllib.request
from bs4 import BeautifulSoup



# coding: utf-8

# In[ ]:


######################################################################################
# THis example is pretty much entirely based on this excellent blog post
# http://glowingpython.blogspot.in/2014/09/text-summarization-with-nltk.html
# Thanks to TheGlowingPython, the good soul that wrote this excellent article!
# That blog is is really interesting btw.
######################################################################################


######################################################################################
# nltk - "natural language toolkit" is a python library with support for
#         natural language processing. Super-handy.
# Specifically, we will use 2 functions from nltk
#
#  sent_tokenize: given a group of text, tokenize (split) it into sentences
#  word_tokenize: given a group of text, tokenize (split) it into words
#  stopwords.words('english') to find and ignored very common words ('I', 'the',...)
#  to use stopwords, you need to have run nltk.download() first - one-off setup
######################################################################################
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords

######################################################################################
# We have use dictionaries so far, but now that we have covered classes - this is a good
# time to introduce defaultdict. THis is class that inherits from dictionary, but has
# one additional nice feature: Usually, a Python dictionary throws a KeyError if you try
# to get an item with a key that is not currently in the dictionary.
# The defaultdict in contrast will simply create any items that you try to access
# (provided of course they do not exist yet). To create such a "default" item, it relies
# a function that is passed in..more below.
######################################################################################
from collections import defaultdict

######################################################################################
#  punctuation to ignore punctuation symbols
######################################################################################
from string import punctuation

######################################################################################
# heapq.nlargest is a function that given a list, easily and quickly returns
# the 'n' largest elements in the list. More below
######################################################################################
from heapq import nlargest


######################################################################################
# Our first class, named FrequencySummarizer
######################################################################################
class FrequencySummarizer:
    # indentation changes - we are now inside the class definition
    def __init__(self, min_cut=0.1, max_cut=0.9):
        # The constructor named __init__
        # THis function will be called each time an object of this class is
        # instantiated
        # btw, note how the special keyword 'self' is passed in as the first
        # argument to each method (member function).
        self._min_cut = min_cut
        self._max_cut = max_cut
        # Words that have a frequency term lower than min_cut
        # or higer than max_cut will be ignored.
        self._stopwords = set(stopwords.words('english') + list(punctuation))
        # Punctuation symbols and stopwords (common words like 'an','the' etc) are ignored
        #
        # Here self._min_cut, self._max_cut and self._stopwords are all member variables
        # i.e. each object (instance) of this class will have an independent version of these
        # variables.
        # Note how this function is used to set up the member variables to their appropriate values
    # indentation changes - we are out of the constructor (member function, but we are still inside)
    # the class.
    # One important note: if you are used to programming in Java or C#: if you define a variable here
    # i.e. outside a member function but inside the class - it becomes a STATIC member variable
    # THis is an important difference from Java, C# (where all member variables would be defined here)
    # and is a common gotcha to be avoided.

    def _compute_frequencies(self, word_sent):
        # next method (member function) which takes in self (the special keyword for this same object)
        # as well as a list of sentences, and outputs a dictionary, where the keys are words, and
        # values are the frequencies of those words in the set of sentences
        freq = defaultdict(int)
        # defaultdict, which we referred to above - is a class that inherits from dictionary,
        # with one difference: Usually, a Python dictionary throws a KeyError if you try
        # to get an item with a key that is not currently in the dictionary.
        # The defaultdict in contrast will simply create any items that you try to access
        # (provided of course they do not exist yet). THe 'int' passed in as argument tells
        # the defaultdict object to create a default value of 0
        for s in word_sent:
        # indentation changes - we are inside the for loop, for each sentence
          for word in s:
            # indentation changes again - this is an inner for loop, once per each word_sent
            # in that sentence
            if word not in self._stopwords:
                # if the word is in the member variable (dictionary) self._stopwords, then ignore it,
                # else increment the frequency. Had the dictionary freq been a regular dictionary (not a
                # defaultdict, we would have had to first check whether this word is in the dict
                freq[word] += 1
        # Done with the frequency calculation - now go through our frequency list and do 2 things
        #   normalize the frequencies by dividing each by the highest frequency (this allows us to
        #            always have frequencies between 0 and 1, which makes comparing them easy
        #   filter out frequencies that are too high or too low. A trick that yields better results.
        m = float(max(freq.values()))
        # get the highest frequency of any word in the list of words

        for w in list(freq.keys()):
            # indentation changes - we are inside the for loop
            freq[w] = freq[w]/m
            # divide each frequency by that max value, so it is now between 0 and 1
            if freq[w] >= self._max_cut or freq[w] <= self._min_cut:
                # indentation changes - we are inside the if statement - if we are here the word is either
                # really common or really uncommon. In either case - delete it from our dictionary
                del freq[w]
                # remember that del can be used to remove a key-value pair from the dictionary
        return freq
        # return the frequency list

    def summarize(self, text, n):
        # next method (member function) which takes in self (the special keyword for this same object)
        # as well as the raw text, and the number of sentences we wish the summary to contain. Return the
        # summary
        sents = sent_tokenize(text)
        # split the text into sentences
        assert n <= len(sents)
        # assert is a way of making sure a condition holds true, else an exception is thrown. Used to do
        # sanity checks like making sure the summary is shorter than the original article.
        word_sent = [word_tokenize(s.lower()) for s in sents]
        # This 1 sentence does a lot: it converts each sentence to lower-case, then
        # splits each sentence into words, then takes all of those lists (1 per sentence)
        # and mushes them into 1 big list
        self._freq = self._compute_frequencies(word_sent)
        # make a call to the method (member function) _compute_frequencies, and places that in
        # the member variable _freq.
        ranking = defaultdict(int)
        # create an empty dictionary (of the superior defaultdict variety) to hold the rankings of the
            # sentences.
        for i,sent in enumerate(word_sent):
            # Indentation changes - we are inside the for loop. Oh! and this is a different type of for loop
            # A new built-in function, enumerate(), will make certain loops a bit clearer. enumerate(sequence),
            # will return (0, thing[0]), (1, thing[1]), (2, thing[2]), and so forth.
            # A common idiom to change every element of a list looks like this:
            #  for i in range(len(L)):
            #    item = L[i]
            #    ... compute some result based on item ...
            #    L[i] = result
            # This can be rewritten using enumerate() as:
            # for i, item in enumerate(L):
            #    ... compute some result based on item ...
            #    L[i] = result
            for w in sent:
                # for each word in this sentence
                if w in self._freq:
                    # if this is not a stopword (common word), add the frequency of that word
                    # to the weightage assigned to that sentence
                    ranking[i] += self._freq[w]
        # OK - we are outside the for loop and now have rankings for all the sentences
        sents_idx = nlargest(n, ranking, key=ranking.get)
        # we want to return the first n sentences with highest ranking, use the nlargest function to do so
        # this function needs to know how to get the list of values to rank, so give it a function - simply the
        # get method of the dictionary
        return [sents[j] for j in sents_idx]
       # return a list with these values in a list
# Indentation changes - we are done with our FrequencySummarizer class!



def text_file_handler(path):

    blurbs = []
    for file in os.listdir(path):

        current_blurb = {"FILE": file,
                         "SOURCE": "",
                         "AGENT": "",
                         "GOAL": "",
                         "DATA": "",
                         "METHODS": "",
                         "RESULTS": "",
                         "COMMENTS": ""}

        file_path = os.path.join(path, file)

        with open(file_path, 'r', encoding = "ISO-8859-1") as myfile:
            file_text = myfile.read().replace('\n', ' ')

            current_blurb["SOURCE"] = find_between(file_text,
                                                   "SOURCE",
                                                   "AGENT").strip()
            current_blurb["AGENT"] = find_between(file_text,
                                                  "AGENT",
                                                  "GOAL").strip()
            current_blurb["GOAL"] = find_between(file_text,
                                                 "GOAL",
                                                 "DATA").strip()
            current_blurb["DATA"] = find_between(file_text,
                                                 "DATA",
                                                 "METHODS").strip()
            current_blurb["METHODS"] = find_between(file_text,
                                                    "METHODS",
                                                    "RESULTS").strip()
            current_blurb["RESULTS"] = find_between(file_text,
                                                    "RESULTS",
                                                    "COMMENTS").strip()
            current_blurb["COMMENTS"] = find_after(file_text,
                                                   "COMMENTS").strip()

            blurbs.append(current_blurb)

    return blurbs


def find_between(text, first, last):
    try:
        start = text.index(first) + len(first)
        end = text.index(last, start)
        return text[start:end]
    except ValueError:
        return ""
    except IndexError:
        try:
            return text.split(first, 1)[1]
        except ValueError:
            return ""


def find_after(text, first):
    try:
        return text.split("COMMENTS", 1)[1]
    except ValueError:
        return ""
    except IndexError:
        return ""


def get_only_text_from_url(url):
    # This function takes in a URL as an argument, and returns only the text of
    # the article in that URL.
    try:
        page = urllib.request.urlopen(url).read().decode('utf8')
    except:
        print("None")
        return(None, None)
    # download the URL
    url_data = BeautifulSoup(page, 'lxml')

    if url_data is None:
        return(None, None)

    #kill all script and style elements
    for script in url_data(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = ' '.join(map(lambda p: p.text, url_data.find_all('p')))

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    return url_data.title.text, text


def get_urls(url_text):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', url_text)
    return urls


if __name__ == "__main__":
    #Desktop
    path = "/home/daire/Desktop/CS401_Projects/Machine_Learning_in_the_Public_Eye/blurbs"
    #Laptop
    #path = "/home/daire/Code/CS401_Projects/Machine_Learning_in_the_Public_Eye/blurbs"
    blurbs = text_file_handler(path)

    for blurb in blurbs:

        urls = get_urls(blurb["SOURCE"])

        print(blurbs[0]["FILE"])

        for url in urls:
            textOfUrl = get_only_text_from_url(url)

            fs = FrequencySummarizer()
            # instantiate our FrequencySummarizer class and get an object of this class
            summary = fs.summarize(textOfUrl[1], 3)
            print("\n\n")
            print(summary)
        print("\n\nNew Blurb\n\n\n")
