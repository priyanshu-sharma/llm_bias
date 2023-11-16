import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def initial_pre():
    '''
    output of above algorithm
    {'federal': {'line_count': 2055, 'total_length': 929768},
    'calstate': {'line_count': 1202, 'total_length': 316729},
    'texasstate': {'line_count': 367, 'total_length': 66249},
    'nystate': {'line_count': 1589, 'total_length': 579297},
    'utahstate': {'line_count': 870, 'total_length': 181657},
    'ucr': {'line_count': 777, 'total_length': 200725},
    'ucb': {'line_count': 736, 'total_length': 132003},
    'utaustin': {'line_count': 1572, 'total_length': 438816},
    'uutah': {'line_count': 4869, 'total_length': 460946},
    'mit': {'line_count': 3566, 'total_length': 1365313},
    'harvard': {'line_count': 862, 'total_length': 182196},
    'mass': {'line_count': 1438, 'total_length': 405073},
    'my': {'line_count': 67, 'total_length': 16989},
    'umd': {'line_count': 3111, 'total_length': 714274},
    'umbc': {'line_count': 1405, 'total_length': 343018},
    'umb': {'line_count': 4939, 'total_length': 795710},
    'scstate': {'line_count': 993, 'total_length': 262531},
    'usc': {'line_count': 2648, 'total_length': 285363},
    'bc': {'line_count': 1649, 'total_length': 335745},
    'kstate': {'line_count': 1253, 'total_length': 216362},
    'uk': {'line_count': 2928, 'total_length': 1542431},
    'ksu': {'line_count': 2031, 'total_length': 938125}}
    '''
    final_count = {}

    df = pd.read_csv('data.csv')
    print(df.head())

    for i in range(len(df)):
        s_no, name, include = df['s_no'][i], df['name'][i], df['include'][i]
        if include:
            f = open("processed/{}/{}.txt".format(name, s_no), "r")
            binary_file = open("final/{}.txt".format(name), "a")
            total_length = 0
            line_count = 0
            for x in f:
                total_length = total_length + len(x)
                line_count = line_count + 1
                binary_file.write(x)
            if name not in final_count.keys():
                final_count[name] = {
                    'line_count': line_count,
                    'total_length': total_length
                }
            else:
                final_count[name]['line_count'] = final_count[name]['line_count'] + line_count
                final_count[name]['total_length'] = final_count[name]['total_length'] + total_length
            f.close()
            binary_file.close()
    print(final_count)
    return final_count


class TextAnalysis:
    def __init__(self):
        self.initial_vector = {}
        self.PUNCT_TO_REMOVE = "!\"#$%&\'()*+,-./:;<=>?@[\\]^_{|}~`"
        self.unique_names = None
        self.STOPWORDS = set(stopwords.words('english'))
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('omw-1.4')
        nltk.download("vader_lexicon")
        self.WORD_THRESHOLD = 20
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.sent_analyzer = SentimentIntensityAnalyzer()

    def initialize_vector(self):
        df = pd.read_csv('data.csv')
        self.unique_names = df.name.unique()
        for name in self.unique_names:
            self.initial_vector[name] = {}
            f = open("final/{}.txt".format(name), "r")
            data = ''
            for text in f:
                data = data + text + ' '
            self.initial_vector[name]['raw_text'] = data
            f.close()
        print("------------Text Extraction-------------")
 
    def convert_lowercase(self, text):
        return text.lower()

    def remove_punctuation_marks(self, text):
        for i in self.PUNCT_TO_REMOVE:
            text = text.translate(str.maketrans('', '', i))
        return text

    def remove_stopwords(self, text):
        return " ".join([word for word in str(text).split() if word not in self.STOPWORDS])

    def remove_urls(self, text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)

    def remove_html(self, text):
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', text)

    def word_counter(self, text):
        cnt = Counter()
        for word in text.split():
            cnt[word] += 1
        most_common = set([(w, wc) for (w, wc) in cnt.most_common(self.WORD_THRESHOLD)])
        least_common = set([(w, wc) for (w, wc) in cnt.most_common()[:-self.WORD_THRESHOLD-1:-1]])
        return most_common, least_common

    def stemming(self, text):
        return " ".join([self.stemmer.stem(word) for word in text.split()])

    def lemmatization(self, text):
        return " ".join([self.lemmatizer.lemmatize(word) for word in text.split()])

    def sentiment_analysis(self, data_without_punc, filtered_text, stem_words, lemma_words):
        wptp = TextBlob(data_without_punc).sentiment.polarity
        wpts = TextBlob(data_without_punc).sentiment.subjectivity
        ftp = TextBlob(filtered_text).sentiment.polarity
        fts = TextBlob(filtered_text).sentiment.subjectivity
        swp = TextBlob(stem_words).sentiment.polarity
        sws = TextBlob(stem_words).sentiment.subjectivity
        lwp = TextBlob(lemma_words).sentiment.polarity
        lws = TextBlob(lemma_words).sentiment.subjectivity
        return wptp, wpts, ftp, fts, swp, sws, lwp, lws

    def get_vader_score(self, raw_text):
        vader_score = self.sent_analyzer.polarity_scores(raw_text)
        vader_neg = vader_score.get('neg', None)
        vader_neu = vader_score.get('neu', None)
        vader_pos = vader_score.get('pos', None)
        vader_compound = vader_score.get('compound', None)
        return vader_neg, vader_neu, vader_pos, vader_compound

    def preprocess(self):
        for name, data in self.initial_vector.items():
            lowercase_data = self.convert_lowercase(data['raw_text'])
            without_urls = self.remove_urls(lowercase_data)
            without_html = self.remove_html(without_urls)
            self.initial_vector[name]['raw_text'] = without_html
            data_without_punc = self.remove_punctuation_marks(without_html)
            self.initial_vector[name]['wp_text'] = data_without_punc
            filtered_text = self.remove_stopwords(data_without_punc)
            self.initial_vector[name]['text'] = filtered_text
            most_common, least_common = self.word_counter(filtered_text)
            self.initial_vector[name]['most_common'] = most_common
            self.initial_vector[name]['least_common'] = least_common
            stem_words = self.stemming(filtered_text)
            self.initial_vector[name]['stem'] = stem_words
            lemma_words = self.lemmatization(filtered_text)
            self.initial_vector[name]['lemma'] = lemma_words
            if name == 'federal':
                self.initial_vector[name]['type'] = 'neutral'
            elif name in ['calstate', 'nystate', 'ucr', 'ucb', 'mit', 'harvard', 'mass', 'my', 'umd', 'umbc', 'umb']:
                self.initial_vector[name]['type'] = 'blue'
            elif name in ['texasstate', 'utahstate', 'utaustin', 'uutah', 'scstate', 'usc', 'bc', 'kstate', 'uk', 'ksu']:
                self.initial_vector[name]['type'] = 'red'
            else:
                raise NotImplementedError
            wptp, wpts, ftp, fts, swp, sws, lwp, lws = self.sentiment_analysis(data_without_punc, filtered_text, stem_words, lemma_words)
            self.initial_vector[name]['wptp'] = wptp 
            self.initial_vector[name]['wpts'] = wpts 
            self.initial_vector[name]['ftp'] = ftp 
            self.initial_vector[name]['fts'] = fts 
            self.initial_vector[name]['swp'] = swp 
            self.initial_vector[name]['sws'] = sws 
            self.initial_vector[name]['lwp'] = lwp 
            self.initial_vector[name]['lws'] = lws
            vader_neg, vader_neu, vader_pos, vader_compound = self.get_vader_score(without_html)
            self.initial_vector[name]['vader_neg'] = vader_neg
            self.initial_vector[name]['vader_neu'] = vader_neu
            self.initial_vector[name]['vader_pos'] = vader_pos
            self.initial_vector[name]['vader_compound'] = vader_compound

    def orchestrate(self):
        self.initialize_vector()
        self.preprocess()

# from prep import TextAnalysis
# ta = TextAnalysis()
# ta.orchestrate()
# print(ta.initial_vector)


