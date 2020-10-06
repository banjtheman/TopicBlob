import logging
import re
from collections import defaultdict
from typing import List

import pandas as pd
from rank_bm25 import BM25Okapi
import nltk
from gensim import corpora
from gensim import models
from gensim import similarities

from gensim.parsing.preprocessing import (
    preprocess_string,
    strip_punctuation,
    strip_numeric,
)

from stop_words import get_stop_words
from operator import itemgetter


word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
my_punctuation = "#!\"$%&'()*+,-./:;<=>?[\\]^_`{|}~•@“…ə"
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def compile_list_of_stopwords(extra: List[str] = None):
    my_stopwords = nltk.corpus.stopwords.words("english")
    my_stopwords.extend(get_stop_words("english"))
    # If a list of extra stopwords is passed in, extend the list.
    if extra:
        my_stopwords.extend(extra)
    return list(set(my_stopwords))


# TODO have uses pass in thier own clean function?
# cleaning master function
def clean_text(text: str, my_stopwords: List[str], bigrams: bool = False):
    text = text.lower()  # lower case
    text = re.sub("[" + my_punctuation + "]+", " ", text)  # strip punctuation
    text = re.sub("\s+", " ", text)  # remove double spacing
    # text = re.sub('([0-9]+)', '', text) # remove numbers
    text_token_list = [
        word for word in text.split(" ") if word not in my_stopwords
    ]  # remove stopwords

    # text_token_list = [word_rooter(word) if '#' not in word else word
    #                     for word in text_token_list] # apply word rooter
    # if bigrams:
    #     text_token_list = text_token_list+[text_token_list[i]+'_'+text_token_list[i+1]
    #                                         for i in range(len(text_token_list)-1)]
    text = " ".join(text_token_list)
    return text


def topic_search(topicblobs, topics):

    topic_list = set(topics.split(" "))
    docs = []

    for key in topicblobs.blobs.keys():
        topicblob = topicblobs.blobs[key]
        topic_set = set(topicblob["topics"])

        if topic_list.intersection(topic_set):
            docs.append(topicblob)

    return docs


def ranked_search(query: str, blobs: dict):
    corpus_docs = [blob["doc"] for blob in blobs.values()]
    tokenized_corpus = [doc.split() for doc in corpus_docs]

    # corpus_docs = []
    # for key in docs.keys():
    #     corpus_docs.append(docs[key]["doc"])
    # print(corpus_docs)

    tokenized_corpus = [doc[0].split(" ") for doc in corpus_docs]
    # TODO: Add option for tokenizer
    # See https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split()

    # enumerate adds index of element in scores list
    doc_scores = enumerate(bm25.get_scores(tokenized_query))
    sorted_scores = sorted(doc_scores, key=itemgetter(1), reverse=True)
    sorted_scores_index = [i for i, score in sorted_scores]

    sorted_blobs = [blobs.get(i) for i in sorted_scores_index]
    return sorted_blobs


def get_sim_docs(doc, sims):

    sim_list = list(sims)[doc]
    simResp = {}

    sorted_sim = sorted(enumerate(sim_list), key=lambda x: x[1], reverse=True)
    for sim in sorted_sim:
        simResp[sim[0]] = sim[1]
    # print(simResp)
    return simResp


def format_topics_sentences_lsi(LsiModel, corpus, topic_blob):
    """
    Extract all the information needed such as most predominant topic assigned to document and percentage of contribution
    LsiModel= model to be used
    corpus = corpus to be used
    texts = original text to be classify (for topic assignment)
    """
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    doc_counter = 0
    for i, row in enumerate(LsiModel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = LsiModel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),
                    ignore_index=True,
                )

                topic_blob[doc_counter]["topics"] = topic_keywords.split(", ")

                # topic_keywords
                doc_counter += 1
            else:
                break
    sent_topics_df.columns = ["Dominant_Topic", "Perc_Contribution", "Topic_Keywords"]
    return sent_topics_df


def do_topic_modeling(
    docs: List[str], num_topics: int, num_words: int, extra_stop_words: List[str] = None
):

    topicResp = {}
    documents = []
    doc_keys = {}
    # strip documents

    my_stopwords = compile_list_of_stopwords(extra=extra_stop_words)

    for doc in docs:
        cleaned_doc = clean_text(str(doc), my_stopwords)
        documents.append(cleaned_doc)
        # TODO: find better way to store this data
        doc_keys[cleaned_doc] = doc

    # stoplist = set('for a of the and to in'.split())
    texts = [
        [word for word in document.lower().split() if word not in my_stopwords]
        for document in documents
    ]

    # TODO: maybe make this a flag?
    # # remove words that appear only once
    # print("remvoing one time words ...")
    # frequency = defaultdict(int)
    # for text in texts:
    #     for token in text:
    #         frequency[token] += 1

    # texts = [[token for token in text if frequency[token] > 1] for text in texts]

    # print("Init model")
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model
    corpus_tfidf = tfidf[corpus]
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)

    # initialize an LSI transformation
    corpus_lsi = lsi_model[
        corpus_tfidf
    ]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

    topic_blob = {}
    lsi_model.print_topics()
    counter = 0
    for doc, as_text in zip(corpus_lsi, documents):
        topic_blob[counter] = {}
        topic_blob[counter]["doc"] = doc_keys[as_text]
        counter += 1

    index = similarities.MatrixSimilarity(lsi_model[corpus])
    topicResp["sims"] = index

    format_topics_sentences_lsi(lsi_model, corpus, topic_blob)

    topicResp["topic_blobs"] = topic_blob

    # index_list = list(index)
    # for indj in index_list:
    #     print(indj)

    # TODO how do we want to handle saves?
    # with open(".topicblob/topics.json", "w") as outfile:
    #     json.dump(topics, outfile)
    # index.save("topicblob/index.index")
    # dictionary.save(".topicblob/dict.pkl")
    # lsi_model.save(".topicblob/model.lsi")
    # corpora.MmCorpus.serialize(".topicblob/corpus.pkl", corpus)
    return topicResp
