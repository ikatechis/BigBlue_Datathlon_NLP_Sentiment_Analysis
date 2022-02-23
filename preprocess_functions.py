import pandas as pd

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('punkt')
from nltk import word_tokenize, pos_tag
nltk.download('omw-1.4')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk.stem.snowball import SnowballStemmer

import string

import emot

def clean_column(col):
    '''
    input --> column as series
    clean from extra space characters, digits, links
    return Series: the clean column
    '''
    rev = col.copy()
    rev = (rev.str.replace(r'\n|\r|\t', ' ', regex=True)
           .str.replace(r'[^a-zA-Z]', ' ', regex=True)
           .str.replace(r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)', ' ', regex=True)
           .str.replace(r'\s+', ' ', regex=True)
           .str.lower()
          )
    
    return rev

def tokenize_clean(col, stem=True, lemmatize=False):
    '''
    tokenize column and remove stopwords and punctuation
    apply stemmer of lemmatizer
    return tokens as a list of lists
    '''
    rev = col.copy()
    tokens = rev.apply(word_tokenize)
    # remove stopwords
    words = stopwords.words("english")
    lambd = lambda s: [word for word in s if word not in words]
    tokens = tokens.apply(lambd)
    # remove punctuation
    import string
    punk = list(string.punctuation)
    lambd = lambda s: [word for word in s if word not in punk + ['...']]
    tokens = tokens.apply(lambd)
    
    if stem:
        tokens = tokens.apply(lambda s: [SnowballStemmer("english").stem(w) for w in s])
                              
    return tokens
    
def find_emoji(df, column):
        #no multiples
        #keep only meaning 
        #df['emoji']=df[column].apply(emot_obj.emoji)
        #df['emoticon']=df[column].apply(emot_obj.emoticons)
        x={}
        emot_obj = emot.core.emot()
        for index,row in df.iterrows():
            ej=set(emot_obj.emoji(row[column])['mean'])
            et=set(emot_obj.emoticons(row[column])['mean'])
            x[row['review_id']]=(list(ej)+list(et))
        emojies = pd.Series(x)
        return pd.DataFrame(emojies).reset_index().rename({'index': 'review_id', 0: 'emoticon'}, axis=1)
    
def preprocess_dataframe(path='../data/train_gr/train.csv'):
    '''
    Load dataframe from path
    Add columns for clean, tokenized and emojies
    return the initial dataframe with extra columns
    '''
    df = pd.read_csv(path)
    df['clean'] = clean_column(df.user_review)
    df['tokens'] = tokenize_clean(df.clean)
    
    # find emojies and merge them into dataframe
    emojies = find_emoji(df, 'user_review')
    df = df.merge(emojies, on='review_id', how='left')
    
    return df