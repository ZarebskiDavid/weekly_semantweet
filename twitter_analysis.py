from textblob import TextBlob
import csv
import nltk
import string
import tweepy
import re
from wordcloud import WordCloud
import unidecode
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy
#import statsmodels.api as sm
#import statsmodels.formula.api as smf


import seaborn as sns
import pyqrcode

from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

plt.style.use('seaborn')

from PIL import Image
import PIL

#df = pd.read_csv("result_libertarian.csv", sep=',') # import data

files =["result_libertarian.csv", "result_libertarian2.csv", "result_libertarian3.csv"]

df = pd.DataFrame()

list_ = []
for file_ in files:
    d = pd.read_csv(file_,sep=',', index_col=None, header=0)
    list_.append(d)

df = pd.concat(list_)

#df = pd.concat([pd.read_csv(f, index_col=0, sep=',',  header=None, axis=1) for f in files], keys=files)

#  db["hashtag"]

def unicodetoascii(text):
    TEXT = (text.
            replace('\\xe2\\x80\\x99', "'").
            replace('\\xc3\\xa9', 'e').
            replace('\\xe2\\x80\\x90', '-').
            replace('\\xe2\\x80\\x91', '-').
            replace('\\xe2\\x80\\x92', '-').
            replace('\\xe2\\x80\\x93', '-').
            replace('\\xe2\\x80\\x94', '-').
            replace('\\xe2\\x80\\x94', '-').
            replace('\\xe2\\x80\\x98', "'").
            replace('\\xe2\\x80\\x9b', "'").
            replace('\\xe2\\x80\\x9c', '"').
            replace('\\xe2\\x80\\x9c', '"').
            replace('\\xe2\\x80\\x9d', '"').
            replace('\\xe2\\x80\\x9e', '"').
            replace('\\xe2\\x80\\x9f', '"').
            replace('\\xe2\\x80\\xa6', '...').
            replace('\\xe2\\x80\\xb2', "'").
            replace('\\xe2\\x80\\xb3', "'").
            replace('\\xe2\\x80\\xb4', "'").
            replace('\\xe2\\x80\\xb5', "'").
            replace('\\xe2\\x80\\xb6', "'").
            replace('\\xe2\\x80\\xb7', "'").
            replace('\\xe2\\x81\\xba', "+").
            replace('\\xe2\\x81\\xbb', "-").
            replace('\\xe2\\x81\\xbc', "=").
            replace('\\xe2\\x81\\xbd', "(").
            replace('\\xe2\\x81\\xbe', ")")

                 )
    return TEXT

def removeEmoj(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return(emoji_pattern.sub(r'', text)) # no emoji


def sub(x):
    return(TextBlob(x).subjectivity)

def pol(x):
    return(TextBlob(x).polarity)

def getNames(x):
    names = []
    for i in x:
        if (i[1] == 'NN' or i[1] == 'NNP' or i[1] == 'NNS' or i[1] == 'NNPS'):
            names.append(i[0])
    return names
    
def getAdj(x):
    adjs = []
    for i in x:
        if (i[1] == 'JJ'):
            adjs.append(i[0])
    return adjs

def GetTypeofTweet(t):     
    if re.match(r'b[\'\"]@', t[:3]):
        return("response")
    if re.match(r'b[\'\"]RT', t[:4]):
        return("retweet")
    else: 
        return("original")

df['subjectivity'] = df.apply (lambda row: sub(row["text"]),axis=1)
df['polarity'] = df.apply (lambda row: pol(row["text"]),axis=1)

#Get relevant non linguistic informations
df['url'] = df['text'].str.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+') # get the urls
df['namedAuthor'] = df['text'].str.findall(r'@\S+') # get the named authors C'EST CE TRUC QUI MERDE
df['hash'] = df['text'].str.findall('(?<=\s)#\w*') # get the hastags

df['TweetType'] = df.apply (lambda row: GetTypeofTweet(row["text"]),axis=1)

df['textGood'] = df['text'].str[2:]
df['textGood'] = df['textGood'].str[:-1]
df['textGood'] = df['textGood'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','')
df['textGood'] = df['textGood'].str.replace('RT','')
df['textGood'] = df['textGood'].str.replace('#','')
df['textGood'] = df['textGood'].str.replace(r'@\S+','')

df['tokenized'] = df.apply (lambda row: nltk.word_tokenize(row["textGood"]),axis=1)
df['posTag'] = df.apply (lambda row: nltk.pos_tag(row["tokenized"]),axis=1)
df['chuncked'] = df.apply (lambda row: nltk.ne_chunk(row["posTag"], binary=True),axis=1)
df['Names'] = df.apply (lambda row: getNames(row["posTag"]),axis=1)
df['Adj'] = df.apply (lambda row: getAdj(row["posTag"]),axis=1)

df['namedAuthor'] = df['namedAuthor'].apply(pd.Series).astype(str)
df['url'] = df['url'].apply(pd.Series).astype(str)
df['hash'] = df['hash'].apply(pd.Series).astype(str)
df['Names'] = df['Names'].apply(pd.Series).astype(str)
df['Adj'] = df['Adj'].apply(pd.Series).astype(str)


dfOriginal = df[(df['TweetType'] == "original") ]
dfResponse = df[(df['TweetType'] == "response") ]
dfRT = df[(df['TweetType'] == "retweet") ]

#df.head()

#print(df.head())

print("DF OK")

def printing():
    out = Image.new("RGB", (1920, 1920), "white")
    sub = Image.open("subjectivity.png")
    word =  Image.open("word.png")
    out.paste(sub, (0,0))
    out.paste(word, (640,0))
    
    out.save('out.png')

# SENTIMENT ANALYSIS

def MostFrequentUrl(d): # get the 9 most current url and print them in a 3*3 layer
    UrlCount = d['url'].value_counts()
    NineMorstFrequent =UrlCount.head(9).index.values
    j = 0
    for i in NineMorstFrequent: 
        print(i) 
        url = pyqrcode.create(i)
        url.eps(str(j)+'.eps', scale=2)
        j= j+1
        print("Done")
    Urls = Image.new("RGB", (600,640), "white")
    draw = PIL.ImageDraw.Draw(Urls)
    placements = [(0,40), (200,40), (400,40), (0,240), (200,240), (400,240), (0,440), (200,440), (400,440)] # positions
    for i in range(9):
        font3 = PIL.ImageFont.truetype("DroidSansMono.ttf", 45)
        draw.text((45.0, 5.0),"Mains Links" ,(15,15,15),font=font3)
        subImage = Image.open(str(i)+".eps")
        #subImage.resize((200,200))                #A REDIMENTIONNER
        Urls.paste(subImage.resize((200,200)), placements[i])
    Urls.save("urls.png", "PNG")

def Sub_corr(d, title, filename):  # subjectivity ~ polarity  /////  polynomial regression
    d = d[(d["polarity"] != 0) & (d["subjectivity"] != 0) & (d["polarity"] != 1) & (d["subjectivity"] != 1)] # subjectivity = 0 polarity = 0 excluded
    
    dPos = d[(d["polarity"] > 0)]
    dNeg = d[(d["polarity"] < 0)]
    
    dSub = d.sample(n=10000)
    Subj_Pol_corr = np.polyfit(d["polarity"], d["subjectivity"], 3, full = True) #model
    e = round(Subj_Pol_corr[1][0],2) # resid – sum of squared residuals of the least squares fit rank – the numerical rank of the scaled Vandermonde matrix sv – singular values of the scaled Vandermonde matrix rcond – value of rcond.
    p = np.poly1d(Subj_Pol_corr[0])
    print(p)
    
    LinPos = scipy.stats.linregress(dPos["polarity"], dPos["subjectivity"])
    LinNeg = scipy.stats.linregress(dNeg["polarity"], dNeg["subjectivity"])
    
    x = np.array([-1 , 1])

    x_fit = np.linspace(x[0], x[-1], 50)
    y_fit = p(x_fit)
    
    plt.figure()
    plt.plot(dSub["polarity"], dSub["subjectivity"], '.b',
             dPos["polarity"], LinPos[0]*dPos["polarity"]+LinPos[1],'r',
             dNeg["polarity"], LinNeg[0]*dNeg["polarity"]+LinNeg[1],'r',)
    plt.axis([-1, 1, 0, 1])
    plt.title(title)
    plt.xlabel('Polarity (pos/neg textual content)')
    plt.ylabel('Subjectivity')
    plt.text(-0.95, .2, 'n: ' +str(len(dNeg.index)))
    plt.text(0.65, .2, 'n: ' +str(len(dPos.index)))
    plt.text(-0.95, .1, 'R-sq: ' +str(round(LinNeg[2]**2,3)),color='r')
    plt.text(0.65, .1, 'R-sq: ' +str(round(LinPos[2]**2,3)),color='r')
    plt.savefig(filename)
    plt.close()


def TypedDistribution(d):                                   # Splter + / -
    d = d[(d["polarity"] != 0) & (d["subjectivity"] != 0)]
    
    dOriginal = d[(d['TweetType'] == "original") ]
    dResponse = d[(d['TweetType'] == "response") ]
    dRT = d[(d['TweetType'] == "retweet") ]
    
    dOriginalPOs = dOriginal[(dOriginal['polarity'] > 0) ]
    dOriginalNEg = dOriginal[(dOriginal['polarity'] < 0) ]
    dResponsePOs = dResponse[(dResponse['polarity'] > 0) ]
    dResponseNEg = dResponse[(dResponse['polarity'] < 0) ]
    dRTPOs = dRT[(dRT['polarity'] > 0) ]
    dRTNEg = dRT[(dRT['polarity'] < 0) ]
    
    fig, (ax0, ax1, ax2) = plt.subplots(3, sharex=True, figsize=(11,12))
    
    ax = sns.kdeplot(dOriginalPOs["polarity"], dOriginalPOs["subjectivity"], cmap="Greens", shade=True, shade_lowest=False, ax=ax0)
    ax = sns.kdeplot(dOriginalNEg["polarity"], dOriginalNEg["subjectivity"], cmap="Reds", shade=True, shade_lowest=False, ax=ax0)
    ax = sns.kdeplot(dResponsePOs["polarity"], dResponsePOs["subjectivity"],cmap="Greens", shade=True, shade_lowest=False, ax=ax1)
    ax = sns.kdeplot(dResponseNEg["polarity"], dResponseNEg["subjectivity"],cmap="Reds", shade=True, shade_lowest=False, ax=ax1)
    ax = sns.kdeplot(dRTPOs["polarity"], dRTPOs["subjectivity"],cmap="Greens", shade=True, shade_lowest=False, ax=ax2)
    ax = sns.kdeplot(dRTNEg["polarity"], dRTNEg["subjectivity"],cmap="Reds", shade=True, shade_lowest=False, ax=ax2)
    
    blue = sns.color_palette("Blues")[-2]
    red = sns.color_palette("Reds")[-2]
    green = sns.color_palette("Greens")[-2]
    
    ax0.text(-1, 0.2, "original", size=16)
    ax1.text(-1, 0.2, "responses", size=16)
    ax2.text(-1, 0.2, "retweet", size=16)
    
    ax2.text(-1, 0.8, "positive tweet", size=16, color=green)
    ax2.text(-1, 0.7, "negative tweet", size=16, color=red)
    
    plt.suptitle('Polarity VS Subjectivity by type of Tweets', fontsize=20)
    
    plt.savefig('tweetType.png')

# SEMANTIC
def corpusify(t):
    t = t.str.cat(sep=' ') # rafiner le text pour virer #, @blabla et url

    t = re.sub(r'http\S+', '', t) # remove url
    t = re.sub(r'@\S+', '', t) # remove usernames
    t = re.sub(r'#', '', t)
    t = re.sub(r'RT', '', t)
    t = re.sub(r'&amp', '', t)
    t = re.sub(r'GIVEAWAY', '', t)
    t = re.sub(r'follow', '', t)
    t = re.sub(r'Follow', '', t)
    t = re.sub(r'b\S+', '', t)
    t = re.sub(r'\\xf0\S+', '', t) # Remove Emoji
    t = re.sub(r'\\xe2\S+', '', t) # Remove special caracters
    t = unicodetoascii(t)
    return(t)

def GetTokens(t):  # get tf-idf
    
    lowers = t.lower()
    #remove the punctuation using the character deletion step of translate
    #no_punctuation = str.maketrans('', '', string.punctuation)
    
    punctuation = '''''!()-[]{};:'"\,<>./?@#$%^&*_~+'''  
    
    no_punctuation = ""  
    for char in lowers:  
        if char not in punctuation:  
            no_punctuation = no_punctuation + char

    tokens = nltk.word_tokenize(no_punctuation)
    filtered = [w for w in tokens if not w in nltk.corpus.stopwords.words('english')]
    tagged = nltk.pos_tag(filtered)
    count = nltk.Counter(tagged)
    count = count.most_common(50)
    words = [i for i in count if (i[0][1] == 'NN' or i[0][1] == 'JJ' or i[0][1] == 'NNS') and i[0][0] != query and i[0][0] != "query" and i[0][0] != "win" and i[0][0] != "chance" and i[0][0] != "week"]
    return words
    
    

def PrintMainLemmes(n, d):
    d['Adj'] = df.apply (lambda row: nltk.WordNetLemmatizer().lemmatize(row["Adj"]),axis=1)
    d['Adj'] = d['Adj'].str.lower()
    d['Names'] = df.apply (lambda row: nltk.WordNetLemmatizer().lemmatize(row["Names"]),axis=1)
    d['Names'] = d['Names'].str.lower()
    
    HashTop = d['hash'].value_counts().head(n)
    NamesTop = d['Names'].value_counts().head(n)
    AdjTop = d['Adj'].value_counts().head(n)
    
    dAdj = d[d['Adj'].isin(AdjTop.index.values)]
    dNames = d[d['Names'].isin(NamesTop.index.values)]
    dHash = d[d['hash'].isin(HashTop.index.values)]
    
    
    fig, (ax0, ax1, ax2) = plt.subplots(3, sharex=True, figsize=(11,12))
    sns.boxplot(x="polarity", y="Names", data=dNames, ax=ax0)
    sns.boxplot(x="polarity", y="hash", data=dHash, ax=ax1)
    sns.boxplot(x="polarity", y="Adj", data=dAdj, ax=ax2)
    plt.suptitle('Polarity of main Lemmes (names, hashtags, adj)', fontsize=20)
    plt.savefig('lemmes.png')
    plt.close()

    
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    tokens = tokenizer.tokenize(t)
    lemmatizer = nltk.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    stopwords = nltk.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwords]
    
    tf = nltk.Counter(tokens)
    
    
    
    
    #tfidf = tf[t] * idf[t]
    #terms_sorted_tfidf_desc = sorted(tfidf.items(), key=lambda x: -x[1])
    #terms, scores = zip(*terms_sorted_tfidf_desc)
    #keywords = terms[:k]
    
    
    return(type(tf))
    

def cloud(text):
        wordcloud = WordCloud().generate(text)
        plt.imshow(wordcloud)
        plt.axis("off")
        wordcloud = WordCloud(max_font_size=50).generate(text)
        plt.figure()
        plt.imshow(wordcloud)
        plt.axis("off")
        plt.savefig('word.png')
        plt.close()

# VIRALITY

def FollowerVir(d):

    dfvir = d[(d['retwc']>1) ]
    dfvirSub = d.sample(n=10000) # subset for ploting

    corrRetwFollo = scipy.stats.spearmanr(dfvir["followers"],dfvir["retwc"]) # Spearman Rank Correlation Coefficient
    #print(type(corrRetwFollo[0]))
    plt.figure()
    plt.plot(dfvir["followers"], dfvir["retwc"], 'r.', )
    plt.axis([0, 3000, 0, 6000])
    plt.title("nbFollowers $\mathcal{R}$ Re-tweet (excl. 0)")
    
    plt.xlabel('nb Followers')
    plt.ylabel('Re-tweeted')
    plt.text(1950, 5000, 'Corr Spearman: ' +str(round(corrRetwFollo[0], 4)))
    plt.text(1950, 4800, 'p: ' +str(corrRetwFollo[1]))
    plt.text(1950, 5500, 'n: ' +str(len(dfvir.index)))
    plt.savefig('corrRetwFoll.png')
    plt.close()

# positivity and rtwc:
def MostAnsweredAuthors(d, n): 
    d = d[(d['TweetType'] == "response") ]
    AuthorsCount = d['namedAuthor'].value_counts()
    MostFrequent =AuthorsCount.head(n).index.values
    dnamedAuthor = d[d['namedAuthor'].isin(MostFrequent)]
    
    aggregated = dnamedAuthor.groupby(['namedAuthor']).aggregate(np.mean)
    
    fig = plt.figure()
    ax = sns.interactplot("polarity", "subjectivity", "liked", aggregated)
    for i in aggregated.index:
        ax.annotate(i, (aggregated["polarity"].ix[i],aggregated["subjectivity"].ix[i]))
    plt.title("Most answered authors")
    plt.savefig('AuthorResponse.png')
    
def MostRetweetedAuthors(d, n): 
    d = d[(d['TweetType'] == "retweet") ]
    AuthorsCount = d['namedAuthor'].value_counts()
    MostFrequent =AuthorsCount.head(n).index.values
    dnamedAuthor = d[d['namedAuthor'].isin(MostFrequent)]
    
    aggregated = dnamedAuthor.groupby(['namedAuthor']).aggregate(np.mean)
    
    fig = plt.figure()
    ax = sns.interactplot("polarity", "subjectivity", "retwc", aggregated)
    for i in aggregated.index:
        ax.annotate(i, (aggregated["polarity"].ix[i],aggregated["subjectivity"].ix[i]))
    plt.title("Most retweeted authors")
    plt.savefig('AuthorRetweet.png')    


def Posi_Rtw(d):
    dfvir = d[(d['retwc']>1) ]
    dfvir = d[(d['retwc']<4000) ]
    dfvir = dfvir[(dfvir['polarity'] != 0) ]
    dfvirSub = d.sample(n=10000) # subset for ploting
    
    plt.figure()
    plt.plot(dfvir["polarity"], dfvir["retwc"], 'g.', )
    plt.axis([-1, 1, 0, 4000])
    plt.title("Polarity (excl. 0) $\mathcal{R}$ 1 < Re-tweet < 4000")
    
    plt.xlabel('Polarity (pos/neg textual content)')
    plt.ylabel('Re-tweeted')
    plt.text(-0.95, 3500, 'n: ' +str(len(dfvir.index)))
    plt.savefig('corrRetwPos.png')
    plt.close()
    
def Semantic(d):
    aggregated = d.groupby(['keyword']).aggregate(np.mean)
    fig = plt.figure()
    ax = aggregated['retwc'].plot(kind="bar", alpha=0.7, color='r');plt.xticks(rotation=70)
    ax2 = ax.twinx()
    ax2.plot(ax.get_xticks(),aggregated['polarity'],marker='o', color='g')
    ax2.plot(ax.get_xticks(),aggregated['subjectivity'],marker='o', color='b')


    corrPolarity = scipy.stats.spearmanr(aggregated["polarity"],aggregated["retwc"])
    corrSubjectivity = scipy.stats.spearmanr(aggregated["subjectivity"],aggregated["retwc"])

    ax.set_title('Effect of main Lemmes')
    ax.set_ylabel(r"Re-Tweeted", color="r")
    #ax2.set_ylabel(r"Polarity", color="g")

    ybox1 = TextArea("Polarity", textprops=dict(color="g", rotation=90,ha='left',va='bottom'))
    ybox2 = TextArea("and ",     textprops=dict(color="k", rotation=90,ha='left',va='bottom'))
    ybox3 = TextArea("Subjectivity ", textprops=dict(color="b", rotation=90,ha='left',va='bottom'))

    ybox = VPacker(children=[ybox1, ybox2, ybox3],align="bottom", pad=0, sep=5)

    anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False, bbox_to_anchor=(1.10, 0.2), 
                                    bbox_transform=ax.transAxes, borderpad=0.)

    ax.add_artist(anchored_ybox)

    plt.text(6, 0.7,  'Corr Spearman: ' +str(round(corrPolarity[0], 4)), color='g')
    plt.text(6, 0.65, 'p: ' +str(corrPolarity[1]), color='g')
    plt.text(6, 0.6,  'Corr Spearman: ' +str(round(corrSubjectivity[0], 4)), color='b')
    plt.text(6, 0.55, 'p: ' +str(corrSubjectivity[1]), color='b')

    #ax.plot(aggregated.index, aggregated["retwc"], '-', label = 'Swdown')
    plt.savefig('semantic.png')
    
def queryInfo(d,query):
    queryInfo = Image.new("RGB", (480,480), "white")
    font = PIL.ImageFont.truetype("DroidSansMono.ttf", 65)
    font3 = PIL.ImageFont.truetype("DroidSansMono.ttf", 35)
    font2 = PIL.ImageFont.truetype("DroidSansMono.ttf", 10)

    draw = PIL.ImageDraw.Draw(queryInfo)
    draw.rectangle(((0,0),(480,480)), fill="black")
    draw.rectangle(((8,8),(472,472)), fill="white")
    draw.text((15.0, 10.0), query,(15,15,15),font=font)
    draw.text((15.0, 90.0),"n: "+ str(len(d.index)) ,(15,15,15),font=font3)
    draw.text((210.0, 140.0),"lang = ENG" ,(15,15,15),font=font3)
    draw.text((15.0, 190.0),str(d.head()),(15,15,15),font=font2)
    queryInfo.save("queryInfo.png", "PNG")
    
###OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO    

query = "libertarian"

queryInfo(df,query)


FollowerVir(df) # => corrRetwFoll.png

Sub_corr(df, "Sentiment Analysis", 'subjectivity.png') # => subjectivity.png

Posi_Rtw(df) # => corrRetwPos.png   récupère la polarity

MostFrequentUrl(df) # => qrcodes

MostAnsweredAuthors(df, 15)

MostRetweetedAuthors(df, 15)

TypedDistribution(df)

PrintMainLemmes(20, df)


text = corpusify(df["text"])
cloud(text)

keywords = GetTokens(text) # => 

#ON EN EST LÀ

def GetCategory(tx): 
    for i in keywords:
        if i[0][0] in str(tx):
            return(i[0][0])
#        else:
#            return("nan")
        

df['keyword'] = df.apply (lambda row: GetCategory(row["text"]),axis=1)

    #df[i[0][0]] = df['text'].str.contains(i[0][0], na=False)
#if df['text'].str.contains(i[0][0], na=False):
#        df["keyword"] = i 




print(df.head())

df.to_csv("computed.csv")

#mean Rtw/keywords

#Semantic(df)

#printing()
