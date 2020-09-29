import logging
from typing import Any, Dict, List

from .gensim_topic_modeling import do_topic_modeling
from .gensim_topic_modeling import get_sim_docs
from .gensim_topic_modeling import ranked_search
from .gensim_topic_modeling import topic_search

class TopicBlob():


    def __init__(self ,docs: List[str],num_topics: int,num_words:int):
        print("I will do topic modeling on "+str(docs))
        try:
            topicResp = do_topic_modeling(docs,num_topics,num_words)
        except Exception as error:
            logging.error(error)
        
        self.topics = topicResp["topics"]
        self.docs = docs
        self.sims = topicResp["sims"]
    

    def get_sim(self,doc_index):
       return get_sim_docs(doc_index,self.sims)
    
    def get_doc(self,doc_index):
        return self.docs[doc_index]
    

    def search_docs_by_topics(self,topic):
        print("Will search docs by topic")
        return topic_search(self.topics,topic,self.docs)
        
    def ranked_search_docs_by_words(self,words):
        print("Will search docs by words")
        return ranked_search(words,self.docs)     





    
        