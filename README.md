
TopicBlob: Simplified Topic Modeling
====================================


`TopicBlob` is a Python 3 library for processing textual data. It provides a simple API for diving into common natural language processing (NLP) taks around topic modeling such as finding similar documents and provide a list of topics givne input text.


Here is a live demo of TopicBlob working on wikipedia pages  
https://share.streamlit.io/banjtheman/topicblob/main/topicblob_st.py


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

Docker Setup
------------

- Ensure you have (docker)[https://www.docker.com/] installed locally.
- Build local Docker Image
    `docker build -t topicblob:local .`
- Run App (Simply runs the `example.py` module)
    `docker run topicblob:local`

