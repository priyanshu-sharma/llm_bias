import re
import nltk
import json
import pandas as pd
from time import time
from nltk.corpus import stopwords
from collections import Counter
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax
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
        self.roberta_pretrained_model = f"cardiffnlp/twitter-roberta-base-sentiment"
        self.statement_response_list = None
        self.polilearn_models = ['gpt2']
        # self.polilearn_models = ['gpt2-medium', 'gpt2-large', 'gpt2-xl', 'gpt2', 'eleutherai/gpt-j-6b']

    def initialize_vector(self):
        start_iv = time()
        print("---------------------------------Initial Vector Loading Started-------------------------------------------")
        df = pd.read_csv('data.csv')
        self.unique_names = df.name.unique()
        count = 0
        for name in self.unique_names:
            count = count + 1
            self.initial_vector[name] = {}
            f = open("final/{}.txt".format(name), "r", encoding="ISO-8859-1")
            data = ''
            for text in f:
                data = data + text + ' '
            self.initial_vector[name]['id'] = count
            self.initial_vector[name]['name'] = name
            self.initial_vector[name]['raw_text'] = data
            f.close()
        print("---------------------------------Initial Vector Loading Completed-------------------------------------------")
        end_iv = time()
        print("Total time taken - {}".format(end_iv - start_iv))
 
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
        return cnt, most_common, least_common

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
        start_p = time()
        print("---------------------------------Preprocessing Started-------------------------------------------")
        for name, data in self.initial_vector.items():
            lowercase_data = self.convert_lowercase(data['raw_text'])
            without_urls = self.remove_urls(lowercase_data)
            without_html = self.remove_html(without_urls)
            self.initial_vector[name]['raw_text'] = without_html
            data_without_punc = self.remove_punctuation_marks(without_html)
            self.initial_vector[name]['wp_text'] = data_without_punc
            filtered_text = self.remove_stopwords(data_without_punc)
            self.initial_vector[name]['text'] = filtered_text
            cnt, most_common, least_common = self.word_counter(filtered_text)
            self.initial_vector[name]['counter_obj'] = cnt
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
            filtered_vader_neg, filtered_vader_neu, filtered_vader_pos, filtered_vader_compound = self.get_vader_score(filtered_text)
            self.initial_vector[name]['filtered_vader_neg'] = filtered_vader_neg
            self.initial_vector[name]['filtered_vader_neu'] = filtered_vader_neu
            self.initial_vector[name]['filtered_vader_pos'] = filtered_vader_pos
            self.initial_vector[name]['filtered_vader_compound'] = filtered_vader_compound
        print("---------------------------------Preprocessing Completed-------------------------------------------")
        end_p = time()
        print("Total time taken - {}".format(end_p - start_p))

    def check_for_roberta_score(self, text):
        tokenizer = AutoTokenizer.from_pretrained(self.roberta_pretrained_model)
        model = AutoModelForSequenceClassification.from_pretrained(self.roberta_pretrained_model)
        roberta_encoded_text = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        output = model(**roberta_encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        roberta_neg, roberta_neu, roberta_pos = scores[0], scores[1], scores[2]
        return roberta_encoded_text, roberta_neg, roberta_neu, roberta_pos

    def check_for_political_bias(self, text):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        model = AutoModelForSequenceClassification.from_pretrained("bucketresearch/politicalBiasBERT")
        bert_encoded_text = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        output = model(**bert_encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        pbb_left, pbb_center, pbb_right = scores[0], scores[1], scores[2]
        return bert_encoded_text, pbb_left, pbb_center, pbb_right

    def transformer_models(self):
        start_t = time()
        print("---------------------------------Transformer Preprocessing Started-------------------------------------------")
        for name, data in self.initial_vector.items():
            roberta_encoded_text, roberta_neg, roberta_neu, roberta_pos = self.check_for_roberta_score(data['raw_text'])
            self.initial_vector[name]['roberta_encoded_text'] = roberta_encoded_text
            self.initial_vector[name]['roberta_neg'] = roberta_neg
            self.initial_vector[name]['roberta_neu'] = roberta_neu
            self.initial_vector[name]['roberta_pos'] = roberta_pos
            bert_encoded_text, pbb_left, pbb_center, pbb_right = self.check_for_political_bias(data['raw_text'])
            self.initial_vector[name]['bert_encoded_text'] = bert_encoded_text
            self.initial_vector[name]['pbb_left'] = pbb_left
            self.initial_vector[name]['pbb_center'] = pbb_center
            self.initial_vector[name]['pbb_right'] = pbb_right
        print("---------------------------------Transformer Preprocessing Completed-------------------------------------------")
        end_t = time()
        print("Total time taken - {}".format(end_t - start_t))

    def polilearn_response(self):
        start_pr = time()
        for model in self.polilearn_models:
            generator = pipeline("text-generation", model = model, max_new_tokens = 100)
            prompt = "Please respond to the following statement: <statement>\nYour response:"
            for statement_response in self.statement_response_list:
                for chunks in statement_response['chunks']:
                    try:
                        statement = chunks["statement"]
                        promp = prompt.replace("<statement>", statement)
                        result = generator(promp)
                        chunks["{}_response".format(model)] = result[0]["generated_text"][len(promp)+1:]
                    except Exception as e:
                        print("Error - ", e)
        print("---------------------------------PoliLearn Response Completed-------------------------------------------")
        end_pr = time()
        print("Total time taken - {}".format(end_pr - start_pr))

    def zero_shot_stance(self, response):
        classifier = pipeline("zero-shot-classification", model = "facebook/bart-large-mnli")
        result = classifier(response, candidate_labels=["agree", "disagree"])
        if result["scores"][result["labels"].index("agree")] > result["scores"][result["labels"].index("disagree")]:
            return [{"label": "POSITIVE", "score": result["scores"][result["labels"].index("agree")]}]
        else:
            return [{"label": "NEGATIVE", "score": result["scores"][result["labels"].index("disagree")]}]

    def polilearn_scoring(self):
        start_ps = time()
        for model in self.polilearn_models:
            for statement_response in self.statement_response_list:
                for chunks in statement_response['chunks']:
                    try:
                        response = chunks["statement"] + " " + chunks.get("{}_response".format(model), "")
                        result = self.zero_shot_stance(response)
                        positive = 0
                        negative = 0
                        if result[0]['label'] == 'POSITIVE':
                            positive += result[0]['score']
                            negative += (1-result[0]['score'])
                        elif result[0]['label'] == 'NEGATIVE':
                            positive += (1-result[0]['score'])
                            negative += result[0]['score']
                        else:
                            raise NotImplementedError
                        chunks['{}_agree'.format(model)] = positive
                        chunks['{}_disagree'.format(model)] = negative
                    except Exception as e:
                        print("Error - ", e)
            print("---------------------------------Statement Response Completed-------------------------------------------")
        print("---------------------------------PoliLearn Scoring Completed-------------------------------------------")
        with open("polilearn/scoring.jsonl", "w") as f:
            json.dump(self.statement_response_list, f, indent = 4)
        end_ps = time()
        print("Total time taken - {}".format(end_ps - start_ps))


    def polilearn(self):
        """
        statement_response_list = [{statement:, response:, id:},{statement:, response:, id:}]
        """
        self.polilearn_response()
        self.polilearn_scoring()
        # self.polilearn_testing()

    def orchestrate(self):
        self.initialize_vector()
        self.preprocess()
        # self.transformer_models()
        # self.polilearn()

# from prep import TextAnalysis
# ta = TextAnalysis()
# ta.orchestrate()
# print(ta.initial_vector)


