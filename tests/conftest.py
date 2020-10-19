import pytest

@pytest.fixture(scope="module", autouse=True)
def download_ntlk_assets():
    import nltk
    nltk.download('stopwords')

@pytest.fixture
def docs():
    text1 = "The titular threat of The Blob has always struck me as the ultimate moviemonster: an insatiably hungry, amoeba-like mass able to penetrate virtually any safeguard, capable of as a doomed doctor chillingly describes it assimilating flesh on contact. Snide comparisons to gelatin be damned, it's a concept with the most devastating of potential consequences, not unlike the grey goo scenario proposed by technological theorists fearful of artificial intelligence run rampant with cells."
    text2 = "Myeloid derived suppressor cells (MDSC) are immature myeloid cells with immunosuppressive activity. They accumulate in tumor-bearing mice and humans with different types of cancer, including hepatocellular carcinoma (HCC)."
    text3 = "Cells are very useful because they are cells. Cell is also a bad guy in DBZ, that was a bad cells"

    return [text1, text2, text3]