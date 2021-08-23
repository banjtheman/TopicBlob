import logging
import re
from pprint import pprint
from typing import List
import ssl
import nltk
import pandas as pd
from gensim import corpora
from gensim import models
from gensim import similarities
from rank_bm25 import BM25Okapi
from stop_words import get_stop_words

word_rooter = nltk.stem.snowball.PorterStemmer(ignore_stopwords=False).stem
my_punctuation = "#!\"$%&'()*+,-./:;<=>?[\\]^_`{|}~•@“…ə"
logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def download_stop_words():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("stopwords")


def compile_list_of_stopwords(extra: List[str] = None) -> list:
    my_stopwords = nltk.corpus.stopwords.words("english")
    my_stopwords.extend(get_stop_words("english"))
    # If a list of extra stopwords is passed in, extend the list.
    if extra:
        my_stopwords.extend(extra)
    return list(set(my_stopwords))


# TODO have uses pass in thier own clean function?
# cleaning master function
def clean_text(text: str, my_stopwords: List[str], bigrams: bool = False) -> str:
    text = text.lower()  # lower case
    text = re.sub("[" + my_punctuation + "]+", " ", text)  # strip punctuation
    text = re.sub("\s+", " ", text)  # remove double spacing
    text = re.sub("([0-9]+)", "", text)  # remove numbers
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


def check_topic_set(row, topic_list: set) -> bool:
    topic_set = set(row["Topics"])
    if topic_list.intersection(topic_set):
        return True
    else:
        return False


def topic_search(df: pd.DataFrame, topics: str) -> pd.DataFrame:
    topic_list = set(topics.split(" "))
    df_mask = df.apply(lambda row: check_topic_set(row, topic_list), axis=1)
    df = df[df_mask]
    return df


def add_ranked_score(row, doc_scores):
    doc_score = doc_scores[row["Document_No"]]
    return doc_score


def ranked_search(query: str, df: pd.DataFrame):
    corpus_docs = [
        text.lower() for text in df["Original Text"]
    ]  # [df["Cleaned Text"] for blob in blobs.values()]

    tokenized_corpus = [doc.split() for doc in corpus_docs]
    # tokenized_corpus = [doc[0].split(" ") for doc in corpus_docs]
    # TODO: Add option for tokenizer
    # See https://github.com/dorianbrown/rank_bm25/blob/master/rank_bm25.py
    bm25 = BM25Okapi(tokenized_corpus)

    # print(tokenized_corpus)
    tokenized_query = query.lower().split()

    # print(tokenized_query)

    # enumerate adds index of element in scores list
    doc_scores = bm25.get_scores(tokenized_query)

    df["ranked_score"] = df.apply(lambda row: add_ranked_score(row, doc_scores), axis=1)
    df = df.sort_values("ranked_score", ascending=False)

    # sorted_blobs = [blobs.get(i) for i in sorted_scores_index]
    return df


def add_sim(row, sim_list):
    sim_score = sim_list[row["Document_No"]]
    return sim_score


def add_sim_doc(row, doc):
    return doc


def get_sim_docs(doc, sims, df):

    sim_list = list(sims)[doc]

    df["sim_score"] = df.apply(lambda row: add_sim(row, sim_list), axis=1)
    df["sim_to_doc"] = df.apply(lambda row: add_sim_doc(row, doc), axis=1)

    df = df.sort_values("sim_score", ascending=False)
    return df


def format_topics_sentences(ldamodel, corpus, texts, og_docs):
    # Init output
    sent_topics_df = pd.DataFrame()
    topics_json = {}

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        topics_json[i] = {}
        topic_list = []
        prop_list = []
        topic_keyword_list = []
        for j, (topic_num, prop_topic) in enumerate(row):
            # if j == 0:  # => dominant topic
            wp = ldamodel.show_topic(topic_num)
            topic_keywords = ", ".join([word for word, prop in wp])
            topic_keywords = topic_keywords.split(", ")

            # print(f"Percentage : {prop_topic}")
            topic_list.append(int(topic_num))
            prop_list.append(float(round(prop_topic, 4)))
            topic_keyword_list.append(topic_keywords)

            if j == 0:
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),
                    ignore_index=True,
                )

        topics_json[i]["topic_list"] = topic_list
        topics_json[i]["percentage_list"] = prop_list
        topics_json[i]["topic_keyword_list"] = topic_keyword_list

        # sent_topics_df = sent_topics_df.append(
        #     pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),
        #     ignore_index=True,
        # )
            # else:
            #     break
    sent_topics_df.columns = ["Dominant_Topic", "Perc_Contribution", "Topic_Keywords"]

    # Add original text to the end of the output
    contents = pd.Series(texts)
    og_docs_df = pd.Series(og_docs)
    sent_topics_df = pd.concat([sent_topics_df, contents, og_docs_df], axis=1)
    return sent_topics_df, topics_json


def do_topic_modeling(
    docs: List[str], num_topics: int, num_words: int, extra_stop_words: List[str] = None
) -> dict:

    topicResp = {}
    documents = []
    doc_keys = {}
    # strip documents
    my_stopwords = compile_list_of_stopwords(extra=extra_stop_words)

    # TODO: maybe make this a flag?
    # # remove words that appear only once
    # print("remvoing one time words ...")
    # frequency = defaultdict(int)
    # for text in texts:
    #     for token in text:
    #         frequency[token] += 1

    # texts = [[token for token in text if frequency[token] > 1] for text in texts]

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

    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda_model = models.ldamodel.LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=100,
        update_every=1,
        chunksize=10,
        passes=10,
        alpha="symmetric",
        iterations=100,
        per_word_topics=True,
    )

    pprint(lda_model.print_topics())
    df_topic_sents_keywords, topics_json = format_topics_sentences(
        ldamodel=lda_model, corpus=corpus, texts=texts, og_docs=docs
    )
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = [
        "Document_No",
        "Dominant_Topic",
        "Topic_Perc_Contrib",
        "Topics",
        "Cleaned Text",
        "Original Text",
    ]

    tfidf = models.TfidfModel(corpus)  # step 1 -- initialize a model
    corpus_tfidf = tfidf[corpus]
    lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=num_topics)
    index = similarities.MatrixSimilarity(lsi_model[corpus])

    topicResp["sims"] = index
    topicResp["df"] = df_dominant_topic
    topicResp["topics"] = lda_model.show_topics(
        num_topics=num_topics, num_words=num_words
    )
    topicResp["topics_json"] = topics_json

    # TODO how do we want to handle saves?
    # with open(".topicblob/topics.json", "w") as outfile:
    #     json.dump(topics, outfile)
    # index.save("topicblob/index.index")
    # dictionary.save(".topicblob/dict.pkl")
    # lsi_model.save(".topicblob/model.lsi")
    # corpora.MmCorpus.serialize(".topicblob/corpus.pkl", corpus)

    return topicResp
