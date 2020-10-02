import streamlit as st
from topicblob import TopicBlob
from SessionState import SessionState
import pandas as pd


session_state = SessionState.get(run_doclist=False, define_parameters=False, run_topicblob=False)
def rerun():
    session_state.run_doclist = False
    session_state.define_parameters = False
    session_state.run_topicblob = False

if st.button('Rerun'):
    rerun()
st.title('TopicBlob')
docs = st.text_area('Please enter with your texts, one per line')
docs = [[doc] for doc in docs.split('\n') if doc != '']
num_topics = st.number_input('Please, enter how much topics you want:', value=5)
num_words = st.number_input('Please, enter how much words you want:', value=5)
run_doclist = st.button('Create doc list')
if not session_state.run_doclist:
    session_state.run_doclist = run_doclist

if session_state.run_doclist:
    df_docs = pd.DataFrame(docs, columns = ['doc'])
    st.write('Here, you can see your docs and the index of them.')
    st.table(df_docs)
    get_topics = st.checkbox('get topics', False)
    search_docs_topics = st.checkbox('search docs by topics', False)
    ranked_search_docs_words = st.checkbox('ranked search docs by words', False)
    find_similar_docs = st.checkbox('find similar docs', False)
    define_function_parameters = st.button('Define parameters')
    if not session_state.define_parameters:
        session_state.define_parameters = define_function_parameters 
    if session_state.define_parameters:

        tb = TopicBlob(docs, num_topics, num_words)
        topics = [topic for blob in tb.blobs.values() for topic in eval(blob['topics'])]

        if search_docs_topics:
            st.subheader('Search docs by topics')
            search_topic = st.selectbox('Select the topic you want to use to search:',
                        list(set(topics)))

        if ranked_search_docs_words:
            st.subheader('Ranked search docs by words')
            word_input_ranked_search = st.text_input('Insert the word you want to use:')

        if find_similar_docs:
            st.subheader('Find similar docs')
            index_sim = st.selectbox('Select the index of the doc you want to find similar docs:',
                        list(df_docs.index))

        run_topicblob = st.button('Run topicBlob')
        if not session_state.run_topicblob:
            session_state.run_topicblob = run_topicblob

    if session_state.run_topicblob:
        if get_topics:
            st.text(topics)

        if search_docs_topics:
            topic_search = tb.search_docs_by_topics(search_topic)

        if ranked_search_docs_words:
            search = tb.ranked_search_docs_by_words(word_input_ranked_search)
            
        if find_similar_docs:
            sims = tb.get_sim(index_sim)
            for sim in sims.keys():
                print(tb.get_doc(sim))