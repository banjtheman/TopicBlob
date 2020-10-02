from topicblob import TopicBlob


text1 = "The titular threat of The Blob has always struck me as the ultimate moviemonster: an insatiably hungry, amoeba-like mass able to penetrate virtually any safeguard, capable of as a doomed doctor chillingly describes it assimilating flesh on contact. Snide comparisons to gelatin be damned, it's a concept with the most devastating of potential consequences, not unlike the grey goo scenario proposed by technological theorists fearful of artificial intelligence run rampant."

text2 = "Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity. They accumulate in tumor-bearing mice and humans with different types of cancer, including hepatocellular carcinoma (HCC)."

docs = [text1, text2]


def main():
    tb = TopicBlob(docs, 5, 5)

    print(tb.blobs[0]["doc"])
    print(tb.blobs[1]["doc"])

    # show sim docs
    sims = tb.get_sim(0)

    for sim in sims.keys():
        print(tb.get_doc(sim))

    search = tb.ranked_search_docs_by_words("with")
    print(search)

    topic_search = tb.search_docs_by_topics("myeloid")

    print(topic_search)


if __name__ == "__main__":
    main()
