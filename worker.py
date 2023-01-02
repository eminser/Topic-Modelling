import pandas as pd
from nltk.corpus import stopwords
from textblob import Word
from spacy.lang.en import English
import PyPDF2
from gensim import corpora
import gensim
import random
# Reading The Article
# nltk.download('stopwords')
# nltk.download('wordnet')

def read_article(article):
    pdfFileObj = open(article, 'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    print(f"Total number of pages: {pdfReader.numPages}")

    paper = ""
    for i in range(pdfReader.numPages):
        pageObj = pdfReader.getPage(i)
        paper = paper + " " + pageObj.extractText()

    pdfFileObj.close()

    start = "abstract"
    end = "references"
    paper = paper.lower()
    paper = paper[paper.index(start):]  # kicking the entrances (before 'abstract' word)
    paper = paper[:paper.index(end)]  # kicking the last part (after acknowledgement/references)
    text = paper.split(".")
    return pd.DataFrame(text, columns=["text"])


def preparetion(paper):
    df = read_article(paper)
    sw = stopwords.words('english')
    df['text'] = df['text'].str.replace(r'[^\w\s]', '', regex=True)  # harf, sayı ve boşluk dışındakileri attık
    df['text'] = df['text'].str.replace(r'\d', '', regex=True)  # sayıları attık
    df['text'] = df['text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    df['text'] = df['text'].apply(lambda x: " ".join(i for i in str(x).split() if i not in sw))

    words = pd.Series(' '.join(df['text']).split()).value_counts()
    drop = words[words == 1]
    df['text'] = df['text'].apply(lambda x: " ".join(i for i in str(x).split() if i not in drop)) # Rarewords

    df = df[df["text"].apply(lambda x: len(x.split()) > 2)].reset_index(drop=True)  # Some Correlations
    return df


def tokenize(text):
    parser = English()
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return [token for token in lda_tokens if len(token) > 1]


def topics(paper, topic_num=10, words=10):
    text_data = []
    df = preparetion(paper)
    for line in df["text"]:
        tokens = tokenize(line)
        text_data.append(tokens)


    dictionary = corpora.Dictionary(text_data)
    corpus = [dictionary.doc2bow(text) for text in text_data]

    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topic_num, id2word=dictionary, passes=15)
    topics = ldamodel.print_topics(num_words=words)

    text = []
    for i in text_data:
        for j in i:
            text.append(j)

    corpus = dictionary.doc2bow(text)
    get_document_topics = ldamodel.get_document_topics(corpus)
    return get_document_topics, topics



thema, topics = topics("makaleler/3.pdf", words=15)

print(thema)
for topic in topics:
    print(topic)


