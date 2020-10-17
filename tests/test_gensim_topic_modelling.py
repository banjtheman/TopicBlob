from topicblob import add_sim_doc, do_topic_modeling


def test_add_sim_doc():
    doc = "some doc"
    result = add_sim_doc('row', doc)
    assert result == doc


def test_do_topic_modelling(docs):
    res = do_topic_modeling(docs=docs, num_topics=5, num_words=5)
    assert "df" in res
    assert "sims" in res