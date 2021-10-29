from topicblob import TopicBlob


text1 = "The titular threat of The Blob has always struck me as the ultimate moviemonster: an insatiably hungry, amoeba-like mass able to penetrate virtually any safeguard, capable of as a doomed doctor chillingly describes it assimilating flesh on contact. Snide comparisons to gelatin be damned, it's a concept with the most devastating of potential consequences, not unlike the grey goo scenario proposed by technological theorists fearful of artificial intelligence run rampant with cells."

text2 = "Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity. They accumulate in tumor-bearing mice and humans with different types of cancer, including hepatocellular carcinoma (HCC)."


text3 = "Cells are very useful because they are cells. Cell is also a bad guy in DBZ, that was a bad cells"

docs = [text1, text2, text3]

extra_stopwords = ["able"]


def main():
    tb = TopicBlob(docs, 5, 5, extra_stopwords)

    # Show topics for each doc
    print("Each topic")
    print(tb.df.iloc[0]["Topics"])
    print(tb.df.iloc[1]["Topics"])
    print(tb.df.iloc[2]["Topics"])

    print("#################DOC EXAMPLE##################")
    print("Showing docs 0 and 1")
    print(tb.df.iloc[0])
    print(tb.df.iloc[1])

    print("#################Sims EXAMPLE##################")
    print("Showing sim docs for doc number 0")
    sims = tb.get_sim(0)
    print(sims)

    print("#################Ranked Search EXAMPLE##################")
    print("Doing ranked search for the word 'with'")
    search = tb.ranked_search_docs_by_words("cells")
    print(search)

    print("#################Topic Search EXAMPLE##################")
    print("Doing topic search for the word 'myeloid'")
    topic_search = tb.search_docs_by_topics("myeloid")
    print(topic_search)


if __name__ == "__main__":
    main()
