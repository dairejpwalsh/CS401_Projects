import os
import re
import urllib.request
from bs4 import BeautifulSoup



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
    url_data = BeautifulSoup(page, 'html.parser')

    if url_data is None:
        return(None, None)

    #kill all script and style elements
    #for script in url_data(["script", "style"]):
        #script.extract()    # rip it out

    # get text
    text = ' '.join(map(lambda p: p.text, url_data.find_all('p')))
    print(text)

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
    # Home Desktop
    path = "/home/daire/Desktop/CS401_Projects/Machine_Learning_in_the_Public_Eye/blurbs"
    # Work Laptop
    path = "/home/daire/Code/CS401_Projects/Machine_Learning_in_the_Public_Eye/blurbs"
    blurbs = text_file_handler(path)

    for blurb in blurbs:

        urls = get_urls(blurb["SOURCE"])

        print(blurbs[0]["FILE"])

        for url in urls:
            textOfUrl = get_only_text_from_url(url)
            print(textOfUrl[0])
            if textOfUrl[1]:
                fs = FrequencySummarizer()
                # instantiate our FrequencySummarizer class and get an object of this class
                summary = fs.summarize(textOfUrl[1], 3)
                print("\n\n")
                print(summary)
        print("\n\nNew Blurb\n\n\n")
