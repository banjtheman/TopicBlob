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

    tb = TopicBlob(texts, 20, 20)

    counter = 0 
    for page in wiki_pages:

        print("Stats for "+page)
        print(tb.df.iloc[counter])
        counter += 1
        print("")

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
