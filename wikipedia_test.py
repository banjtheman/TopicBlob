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


wiki_pages = ["Facebook", "New York City", "Barack Obama"]

texts = []
for page in wiki_pages:
    text = wikipedia.summary(page)
    # print(text)
    texts.append(text)


tb = TopicBlob(texts, 20, 50)


# Do topic search for social

topic_search = tb.search_docs_by_topics("social")
print(topic_search)
print("\n")


# Do a ranked search for president
search = tb.ranked_search_docs_by_words("president")
print(search)
print("\n")

# Find similar text for

print("Finding similar document for\n" + tb.blobs[0]["doc"])
print("\n")
sims = tb.get_sim(0)

for sim in sims.keys():
    print(tb.get_doc(sim))
