import os
import re
import urllib.request
from bs4 import BeautifulSoup
import csv
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd


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
    soup = BeautifulSoup(page, 'lxml')

    if soup is None:
        return(None, None)

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = ' '.join(map(lambda p: p.text, soup.find_all('p')))

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    text_title = ""

    if soup.title is None:
        text_title = ""
    else:
        text_title = soup.title.text

    return [text_title, text]


def get_urls(url_text):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', url_text)
    return urls


def preprocessor(text):

    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)',
                            text)
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text


def topic_grouper(path):
    df = pd.read_csv(path, encoding='utf-8')


    count = CountVectorizer(stop_words='english',
                            strip_accents="unicode",
                            max_df=0.1,
                            lowercase=True,
                            max_features=5000)


    X = count.fit_transform(df[2].values)

    lda = LatentDirichletAllocation(n_components=5,
                                    random_state=1,
                                    learning_method='batch',
                                    n=-1)
    X_topics = lda.fit_transform(X)

    n_top_words = 5
    count = 0
    feature_names = count.get_feature_names()
    for topic in lda.components_:
        count += 1
        print("Topic : " + str(count))

        #print(" ".join([feature_names[i]
        #             for i in topic.argsort()[:-n_top_words - 1:-1]]))

if __name__ == "__main__":
    # Home Desktop
    path = "/home/daire/Desktop/CS401_Projects/Machine_Learning_in_the_Public_Eye/blurbs"
    # Work Laptop
    path = "/home/daire/Code/CS401_Projects/Machine_Learning_in_the_Public_Eye/blurbs"
    blurbs = text_file_handler(path)

    blurbs_to_group = []
    count = 0
    """for blurb in blurbs:

        urls = get_urls(blurb["SOURCE"])

        print(blurb["FILE"])

        # Use only first supplied url if any
        if len(urls) > 0:
            url = urls[0]
            textOfUrl = get_only_text_from_url(url)

            if textOfUrl[0] is not None:
                count+=1
                print(blurb["FILE"])
                print(type(blurb["FILE"]))
                temp_array = [blurb["FILE"], textOfUrl[0], textOfUrl[1]]
                print(temp_array)
                blurbs_to_group.append(temp_array)
                print(type(textOfUrl))
            else:
                print("Bad result")

        print("Next Blurb")
        if count > 10000000:
            break

    with open('points.csv', 'w') as myfile:
        writer = csv.writer(myfile)
        writer.writerows(blurbs_to_group)"""
    topic_grouper("points.csv")
