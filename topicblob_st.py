# Wikipeida topic blob example

import random
import wikipedia
import streamlit as st
import pandas as pd
from topicblob import TopicBlob

wiki_pages_default = [
    "Facebook(Company)",
    "Barack Obama",
    "Wikipedia",
    "Topic Modeling",
    "Python (programming language)",
    "Snapchat",
]


@st.cache(allow_output_mutation=True)
def cache_wiki_pages():

    wiki_pages = [
        "Facebook(Company)",
        "Barack Obama",
        "Wikipedia",
        "Topic Modeling",
        "Python (programming language)",
        "Snapchat",
    ]
    return wiki_pages


@st.cache(allow_output_mutation=True)
def update_wiki_pages(new_pages):

    return new_pages


# Default pages
def select_wiki_pages():
    wiki_pages = cache_wiki_pages()

    # TODO let user add to wiki pages
    st.subheader("Add a wikipedia page for example 'United States' ")

    wiki_page = st.text_input("Type wikipeida page")

    if st.button("Add to page list"):
        wiki_pages.append(wiki_page)

    st.subheader("Reset list to defaults")
    if st.button("Reset list"):
        wiki_pages = wiki_pages_default

    st.subheader("Selected wikipedia pages")
    all_pages = st.multiselect("Select pages", wiki_pages, default=wiki_pages)

    wiki_pages = update_wiki_pages(all_pages)

    return all_pages


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def cahce_wiki_pages(wiki_pages):
    texts = []
    for page in wiki_pages:
        try:
            text = wikipedia.summary(page)
            texts.append(text)
        except wikipedia.DisambiguationError as e:
            alt_page = random.choice(e.options)
            st.warning(f"Picking random page from {e.options}")
            st.warning(f"Using {alt_page}")
            try:
                text = wikipedia.summary(alt_page)
                texts.append(text)
            except wikipedia.DisambiguationError as e2:
                st.error("Be more specific with your Page name")
                st.error(e2.options)

        except wikipedia.PageError as e:
            st.error(e)
            st.error("Be more specific with your Page name")

    return texts


@st.cache(allow_output_mutation=True)
def cahce_topic_blob(texts, num_topics, num_words):
    tb = TopicBlob(texts, num_topics, num_words)
    return tb


def show_ranked_search(row, wiki_pages):
    doc_num = row["Document_No"]
    st.write(f"{wiki_pages[doc_num]}: {row['ranked_score']}")


def show_sim_search(row, wiki_pages, expander):
    doc_num = row["Document_No"]
    expander.write(f"{wiki_pages[doc_num]}: {row['sim_score']}")


def show_topic_search(row, wiki_pages):
    doc_num = row["Document_No"]
    st.write(f"{wiki_pages[doc_num]}: {row['Topics']}")


def main():
    st.title("TopicBlob")
    st.header("Get topics from text")
    st.write(
        "This example will allow you to run TopicBlob on wikipedia pages to show off what it can do"
    )

    wiki_pages = select_wiki_pages()

    # st.write(wiki_pages)

    texts = cahce_wiki_pages(wiki_pages)
    num_topics = st.number_input("Number of topics", min_value=1, value=20, step=1)
    num_words = st.number_input(
        "Number of words per topic", min_value=1, value=20, step=1
    )
    tb = cahce_topic_blob(texts, num_topics, num_words)

    st.write("Here is the topic blob df")
    st.write(tb.df)

    counter = 0
    for page in wiki_pages:

        expander = st.expander(page)
        curr_text = tb.df.iloc[counter]["Original Text"]
        topics = tb.df.iloc[counter]["Topics"]

        expander.header("Original Text")
        expander.text(curr_text)

        expander.header("Topic List")
        expander.write(topics)

        expander.header("These are the most similar pages")
        sims = tb.get_sim(counter)
        # st.write(sims)

        sims.apply(lambda row: show_sim_search(row, wiki_pages, expander), axis=1)
        counter += 1

    st.header("Ranked Search")
    st.subheader(
        "With the docs you can do a ranked word search which will find the documents that mention your input words based on the BM25 algorithm (Note: search IS Case Sensitive)"
    )
    ranked_search_word = st.text_input("Do a ranked search")

    if st.button("Ranked Search"):
        search_results = tb.ranked_search_docs_by_words(ranked_search_word)
        st.write(search_results)
        # for the df show the doc
        search_results.apply(lambda row: show_ranked_search(row, wiki_pages), axis=1)

    st.header("Topic Search")
    st.subheader(
        "With the docs you can do a topic word search which will find the documents that have your topic words (Note: search IS Case Sensitive)"
    )
    topic_search_word = st.text_input("Do a topic search")

    if st.button("Topic Search"):
        search_results = tb.search_docs_by_topics(topic_search_word)
        # st.write(search_results)
        # for the df show the doc
        search_results.apply(lambda row: show_topic_search(row, wiki_pages), axis=1)


if __name__ == "__main__":
    main()
