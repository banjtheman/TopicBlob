import logging
from typing import Any, Dict, List


import pandas as pd  # type: ignore
from .gensim_topic_modeling import do_topic_modeling, download_stop_words
from .gensim_topic_modeling import get_sim_docs
from .gensim_topic_modeling import ranked_search
from .gensim_topic_modeling import topic_search


class TopicBlob:
    def __init__(
        self,
        docs: List[str],
        num_topics: int,
        num_words: int,
        extra_stop_words: List[str] = None,
    ):
        try:
            download_stop_words()
            topicResp = do_topic_modeling(docs, num_topics, num_words, extra_stop_words)
        except Exception as error:
            logging.error(error)

        self.df = topicResp["df"]
        self.sims = topicResp["sims"]
        self.topics = topicResp["topics"]
        self.topics_json = topicResp["topics_json"]

    def get_sim(self, doc_index: int) -> pd.DataFrame:
        return get_sim_docs(doc_index, self.sims, self.df)

    def get_doc(self, doc_index: int) -> pd.Series:
        return self.df.iloc[doc_index]

    def search_docs_by_topics(self, topics: str) -> pd.DataFrame:
        return topic_search(self.df, topics)

    def ranked_search_docs_by_words(self, words: str) -> pd.DataFrame:
        return ranked_search(words, self.df)

    def show_topics(self) -> List[Any]:
        return self.topics
