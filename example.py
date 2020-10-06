from topicblob import TopicBlob


text1 = "The titular threat of The Blob has always struck me as the ultimate moviemonster: an insatiably hungry, amoeba-like mass able to penetrate virtually any safeguard, capable of as a doomed doctor chillingly describes it assimilating flesh on contact. Snide comparisons to gelatin be damned, it's a concept with the most devastating of potential consequences, not unlike the grey goo scenario proposed by technological theorists fearful of artificial intelligence run rampant."

text2 = "Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity. They accumulate in tumor-bearing mice and humans with different types of cancer, including hepatocellular carcinoma (HCC)."

docs = [text1, text2]

extra_stopwords = ["able"]


def main():
    tb = TopicBlob(docs, 5, 5, extra_stopwords)

    print("#################DOC EXAMPLE##################")
    print("Showing docs 0 and 1")
    print(tb.blobs[0]["doc"])
    print(tb.blobs[1]["doc"])

    print("#################Sims EXAMPLE##################")
    print("Showing sim docs for doc number 0")
    sims = tb.get_sim(0)
    for sim in sims.keys():
        print(tb.get_doc(sim))

    print("#################Ranked Search EXAMPLE##################")
    print("Doing ranked search for the word 'with'")
    search = tb.ranked_search_docs_by_words("with")
    print(search)

    print("#################Topic Search EXAMPLE##################")
    print("Doing topic search for the word 'myeloid'")
    topic_search = tb.search_docs_by_topics("myeloid")
    print(topic_search)


if __name__ == "__main__":
    main()
