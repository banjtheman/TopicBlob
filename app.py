import streamlit as st
from topicblob import TopicBlob
from SessionState import SessionState
import pandas as pd


session_state = SessionState.get(run_doclist=False, define_parameters=False, run_topicblob=False, topicblob=None)
def rerun():
    session_state.run_doclist = False
    session_state.define_parameters = False
    session_state.run_topicblob = False
    session_state.topicblob = None

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

        if session_state.topicblob is None:
            session_state.topicblob = TopicBlob(docs, num_topics, num_words)
        
        topics = [topic for blob in session_state.topicblob.blobs.values() for topic in eval(blob['topics'])]

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
            st.subheader('Topics')
            st.table(pd.DataFrame([[index, blob['topics']] for index, blob in enumerate(session_state.topicblob.blobs.values())],
                        columns=['Index', 'Topics']).set_index('Index', drop=True))

        if search_docs_topics:
            topic_search = session_state.topicblob.search_docs_by_topics(search_topic)
            st.subheader('Similar docs by topic')
            st.table(pd.DataFrame([[list(doc.values())[0][0], list(doc.values())[1]] for doc in  topic_search],
                     columns=['Docs', 'Topics']))

        if ranked_search_docs_words:
            search = session_state.topicblob.ranked_search_docs_by_words(word_input_ranked_search)
            st.subheader('Ranked Search')
            st.table(pd.DataFrame(search, columns=['Docs']))
            
        if find_similar_docs:
            sims = session_state.topicblob.get_sim(index_sim)
            st.subheader('Similarity Docs by index')
            st.text('Search index: ' + str(index_sim))
            st.table(pd.DataFrame([[key, value] for key, value in sims.items()],
                     columns=['Index', 'Similarity']).sort_values(by='Similarity', ascending=False))