import logging
import re
from collections import defaultdict


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

# TODO have abilty to add stopwords
my_stopwords = nltk.corpus.stopwords.words("english")
word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
my_punctuation = "#!\"$%&'()*+,-./:;<=>?[\\]^_`{|}~•@“…"


# TODO have uses pass in thier own clean function?
# cleaning master function
def clean_text(text, bigrams=False):
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


def topic_search(topicblobs, topics, topic_docs):

    topic_list = set(topics.split(" "))
    docs = {}

    counter = 0

    for topicblob in topicblobs:

        topic_set = set(eval(topicblob))
        if topic_list.intersection(topic_set):
            topicResp = {}
            topicResp["topics"] = topicblob
            topicResp["doc"] = topic_docs[counter]

            docs[counter] = topicResp

        counter += 1

    return docs


def ranked_search(query, docs):

    tokenized_corpus = [doc.split(" ") for doc in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.split(" ")
    doc_scores = bm25.get_scores(tokenized_query)

    searchResp = {}

    counter = 0

    for score in doc_scores:
        searchResp[counter] = score
        counter += 1

    # sort by highest
    sorted_scores = {
        k: v
        for k, v in sorted(searchResp.items(), reverse=True, key=lambda item: item[1])
    }

    return sorted_scores


def get_sim_docs(doc, sims):

    sim_list = list(sims)[doc]
    simResp = {}

    sorted_sim = sorted(enumerate(sim_list), key=lambda x: x[1], reverse=True)
    for sim in sorted_sim:
        simResp[sim[0]] = sim[1]
    # print(simResp)
    return simResp


def do_topic_modeling(docs, num_topics, num_words):

    topicResp = {}
    documents = []
    # strip documents

    for doc in docs:
        documents.append(clean_text(str(doc)))

    # stoplist = set('for a of the and to in'.split())
    print("stopword scanning..")
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

    print("Init model")
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model

    corpus_tfidf = tfidf[corpus]

    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)

    # TODO: do we need this?
    # initialize an LSI transformation
    # corpus_lsi = lsi_model[
    #     corpus_tfidf
    # ]  # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi

    print("Model done")

    lda_topics = lsi_model.show_topics(num_words=num_words)

    topics = {}
    filters = [lambda x: x.lower(), strip_punctuation, strip_numeric]

    for topic in lda_topics:
        topic_name = str(preprocess_string(topic[1], filters))
        topics[topic_name] = int(topic[0])

    topicResp["topics"] = topics

    index = similarities.MatrixSimilarity(lsi_model[corpus])
    topicResp["sims"] = index

    # TODO how do we want to handle saves?
    # with open(".topicblob/topics.json", "w") as outfile:
    #     json.dump(topics, outfile)
    # index.save("topicblob/index.index")
    # dictionary.save(".topicblob/dict.pkl")
    # lsi_model.save(".topicblob/model.lsi")
    # corpora.MmCorpus.serialize(".topicblob/corpus.pkl", corpus)
    return topicResp
