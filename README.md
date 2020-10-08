
TopicBlob: Simplified Topic Modeling
====================================


`TopicBlob` is a Python 3 library for processing textual data. It provides a simple API for diving into common natural language processing (NLP) taks around topic modeling such as finding similar documents and provide a list of topics givne input text.


```
import wikipedia
from topicblob import TopicBlob


# get random wikipeida summaries

wiki_pages = [
    "Facebook",
    "New York City",
    "Barack Obama",
    "Wikipedia",
    "Topic Modeling",
    "Python (programming language)",
    "Snapchat",
]


def main():
    texts = []
    for page in wiki_pages:
        text = wikipedia.summary(page)
        # print(text)
        texts.append(text)

    tb = TopicBlob(texts, 5, 5)

    print("#################Sims EXAMPLE##################")
    print("Showing sim docs for doc number 0 *Facebook")
    sims = tb.get_sim(0)
    print(sims)

    print("#################Ranked Search EXAMPLE##################")
    print("Doing ranked search for the word 'president'")
    search = tb.ranked_search_docs_by_words("president")
    print(search)

    print("#################Topic Search EXAMPLE##################")
    print("Doing topic search for the word 'python'")
    topic_search = tb.search_docs_by_topics("python")
    print(topic_search)

if __name__ == "__main__":
    main()

```  

TopicBlob leverages  `NLTK`, `pandas`, and `gensim` , for the heavy lifting

Features
--------

- Topic Extraction
- Similarity Search
- BM25 search ( word ranking search)
- Topic Search

Get it now
----------
    #TODO push to pip
    $ git clone https://github.com/banjtheman/TopicBlob/
    $ pip install --editable . 

Requirements
------------

- Python  >= 3.5


