# NLTK

## Introduction

```python
import nltk
nltk.download() ## download directory 지정
nltk.__file__ ## nltk 설치 파일 위치
```

- 실습

```python
from nltk.book import * # 9 texts from nltk.book 

# text 크기
texts = [text1, text2, text3]
for text in texts:
	print(text, len(text))

# sent 출력
sents = [sent1, sent2, sent3]
for sent in sents:
	print(len(sent), sent)

# concordance
text1.concordance("word")
```

- Dispersion plot

```python
# dispersion plot
text1.dispersion_plot(["word1", "word2"])

## 임의로 지정한 n개의 단어에 대한 dispersion plot
wordList = []
for i in range(n):
	word = input("Enter a word:")
	wordList.append(word)

for text in texts:
	print(text)
	text.dispersion_plot(wordList)
```

- Lexical diversity & percentage

```python
def lexial_diversity(text):
	return(len(set(text))/len(text))

def percentage(count, total):
	return(100*count/total)
```

- NLTK 내장 함수

```python
# FreqDist()
fdist1 = FreqDist(text1)
fdist1.max() ## 가장 많이 나타나는 단어
fdist1.most_common(n) ## 자주 나타나는 상위 n개의 단어
fdist1["word"] ## word의 frequency
fdist1.N() ## total number of samples
fdist1.keys*( ## 내림차순으로 정렬된 sample
fdist1.plot() ## frequency distribution
fdist1.plot(cumulative=True) ## cumulative plot

## 예시 : 8글자 이상이고 8번 이상 나타나는 단어 sort
fdist = FreqDist(text)
sorted(w for w in set(text) if len(w) > 7 and fdist[w] > 7)
```

- n-gram : unigram (n=1), bigram (n=2), trigram (n=3)

 ** contiguous sequence of n items from a given sample of text 

( * collocation : A sequence of words or terms that co-occur more often than would be expected by chance)

```python
list(nltk.bigrams(["A", "B", "C", "D"]))
text.collocations()
```

- Generator expression과 sorted 함수

```python
# 특정 글자로 끝나는 단어
sorted(w for w in set(text) if w.endswith("abc")

# 특정 글자를 포함하는 단어
sorted(term for term in set(text) if "abc" in term)

# 첫 글자가 대문자이고 나머지 글자는 소문자인 단어
sorted(item for iterm in set(text) if item.istitle())

# 숫자
sorted(item for item in set(sent) if item.isdigit())
```

## Corpus

: python에서 `nltk.corpus`를 import 하면, NLTK data distribution에서 corpora에 접근하는 데 사용되는 corpus reader instances 집합이 자동으로 생성

```python
import nltk
from nltk.corpus import *

brown; treebank; names; genesis; inaugural
```

- raw text, a list of words, a list of sentences, or a list of paragraphs

```python
import nltk
from nltk.corpus import inaugural 
inaugural.raw('1789-Vashington.txt')
inaugural.words('1789-Washington.txt')
inaugural.sents('1789-Washington.txt')
inaugural.paras('1789-Washington.txt')
```

- 데이터에 포함된 파일 분석

** fileids() : The files of the corpus

** fileids([categories]) : The files of the corpus corresponding to these categories

** categories() : The categories of the corpus

** categories([fileids]) : The categories of the corpus corresponding to these files

** raw() : The raw content of the corpus

** raw(fileids=f1, f2, f3]) : The raw content of the specified files

** raw(categories=[c1, c2]) : The raw content of specified categories
	→ words(), sents()에서 같은 방식으로 적용

** abspath(fileid) : The location of the given file on disk

** encoding(fileid) : The encoding of the file (if known)

** open(fileid) : Open a stream for reading the given corpus file

** root() : The path to the root of locally installed corpus

** readme() : The contents of the README file of the corpus

```python
from nltk.corpus import gutenberg

# 파일 이름
gutenberg.fileids()

# 각 파일별 길이
for fileid in gutenberg.fileids():
	num_chars = len(gutenberg.raw(fileid))
	num_words = len(gutenberg.words(fileid))
	num_sents = len(gutenberg.sents(fileid))
	num_vocab = len(set(w.lower() for w in gutenberg.words(fileid)))
	print(round(num_chars/num_words), round(num_words/num_sents), round(num_words/num_vocab), fileid)
```

```python
# Web text
from nltk.corpus import webtext

for fileid in webtext.fileids():
	print(fileid, webtext.raw(fileid)[:30], '...')

# Chat text (a corpus of instant messaging chat sessions)
from nltk.corpus import nps_chat

nps_chat.fileids()
chatroom = nps_chat.posts('10-19-30s_705posts.xml')
chatroom[100]

# Brown Corpus : Genre
brown.words(categories='news')
brown.words(fileids=['cg22'])
brown.sents(categories=['news', 'editorial'])
brown.fileids()
	## stylistics 분석
news_text = brown.words(categories='news')
fd = nltk.FreqDist(w.lower() for w in news.text)
m = ['can', 'could', 'may']
for word in m:
	print(word, fd[word])
```

- Conditional Frequency Distribution : A collection of frequency distribution, each one for a different "condition".

```python
cfd = nltk.ConditionalFreqDist(
	(genre, word)
	for genre in brown.categories()
	for word in brown.words(categories=genre))

genres = ['news', 'religion']
modals = ['can', 'could']

cfd.tabulate(conditions=genres, samples=modals)
```

- 연도 별 추이 분석

```python
from nltk.corpus import inaugural

x = [(target, fileid[:4])
	for fileid in inaugural.fileids()
	for w in inaugural.words(fileid)
	for target in ['america', 'citizen']
	if w.lower().startswith(target)]
len(x)
list(x)

# plot
cfd = nltk.ConditionalFreqDist(
				(target, fileid[:4])
				for fileid in inqugural.fileids()
				for w in inaugural.words(fileid)
				for target in ['america', 'citizen']
				if w.lower().startswith(target))
cfd.plot()
```

- Corpora in Other Languages

```python
from nltk.corpus import *

nltk.corpus.cess_esp.words()
nltk.corpus.floresta.words()
nltk.corpus.indian.words('hindi.pos')
nltk.corpus.udhr.words('Korean_Hankuko-UTF8')
nltk.corpus.udhr.words('Japanese_Nihongo-UTF8')

nltk.corpus.udhr.fileids()
```

- 한글 형태소 분석 : KoNLPy
- Plaintext Corpus Reader : 사용자 말뭉치 사용

```python
from nltk.corpus import PlaintextCorpusReader

corpus_root = '/usr/share/dict'
wordlists = PlaintextCorpusReader(corpus_root, '.*')
wordlists.fileids()
wordlists.words('connectives')
```

```python
# Genomics & Informatics 논문지의 corpus 만들기
from nltk.corpus import *

corpus_root = "C://nltk_data/GISampleCorpus/"
GNICorpus = PlaintextCorpusReader(corpus_root, ".*\/txt", encoding="utf-8")
GNICorpus.fileids()
GNICorpus.raw('gni-1-1-32.txt')

 ## Reading data
import nltk
from nltk.corpus import *
import os

corpus_root = os.getwcd() + "/GISampleCorpus"
GenomInfoCorpus = PlaintextCorpusReader(corpus_root, ".*\.txt", encoding = "utf-8")
GenomInfoCorpus.fileids()
giWords = GenomIfoCorpus.words()
len(giWords()
giWords[1:20]

## raw, words, sents
rawString = GenomInfoCorpus('gni-1-1-25.txt')
rawString[1:200]

wordList = GenomInfoCorpus.words('gni-1-1-25.txt')
wordList[1:20]

sentList = GenomInfoCorpus.sents('gni-1-1-25.txt')
sentList[20]
```

- Bracket Parse Corpus Reader

```python
from nltk.corpus import BracketParseCorpusReader

corpus_root = r'C:\corpora\penntreebank\parsed\mrg\wsj'
file_pattern = r'./wsj_.*\.mrg'
ptb = BracketParseCorpusReader(corpus_root, file_pattern)
ptb.fileids()
len(ptb.sents())
ptb.sents(fileids='20/wsj_2013.mrg')[19]
```

- Conditional Frequency Distributions : process a sequence of (condition, event) pairs

```python
pairs = [('condition1', 'event1'), ('condition1', 'event2'), ('condition2', 'event1')]

# Create a conditional frequency distribution from a list of pairs
cfdist = ConditionalFreqDist(pairs)

# Alphabetically sorted list of conditions
cfdist.conditions()

# The frequency distribution for this condition
cfdist[condition]

# Frequency for the given sample for this condition
cfdist[condition][sample]

# Tabulate the conditional frequency distribution
cfdist.tabulate()

# Tabulation limited to the specified samples and conditions
cfdist.tabulate(samples, conditions)

# Graphical plot of the conditional frequency distribution
cfdist.plot()

# Graphical plot limited to the specified samples and coditions
cfdist.plot(samples, conditions)

# Test if samples in cfdist1 occur less freqeuntly than in cfdist2
cfdist1 < cfdist2
```

- Bigrams → random text 생성

```python
text = nltk.corpus.gutenberg.words('austen-emma.txt')
bigrams = nltk.bigrams(text)
cfd = nltk.ConditionalFreqDist(bigrams)
cfd['living']
```

- Lexial Resources : 언어 지식 능력 측정

** Stop words : 추가 처리 전, filtering 할 high frequency words (e.g. the, to) 

```python
from nltk.corpus import stopwords

stopwords = stopwords.words('english') # nltk.corpus.stopwords.words('english')
reuter = nltk.corpus.reuters.words()
filteredReuter = [w for w in reuter if w.lower() not in stopwords]

len(reuter)
len(filteredReuter)

```

ex) 여성, 남성 모두에서 나타나는 이름 찾기

```python
male_names = names.words('male.txt')
female_names = names.words('female.txt')
[w for w in male_names if w in female_names]
```

ex) 성별 이름의 끝 자리 분포 

```python
cfd = nltk.ConditionalFreqDist(
				(fileid, name[-1])
				for fileid in names.fileids()
				for name in names.words(fileid) )
cfd.plot()
```

- [WordNet](http://wordnetweb.princeton.edu/perl/webwn) : lexical database for the English language (동의어, 유의어, 상위어 찾을 수 있음)

```python
# Synonyms, Synset (synonym set), Lemmas (a collection of synonymous words)
from nltk.corpus import wordnet as wn

wn.synsets('motorcar')
wn.synset('car.n.01').lemmas
wn.synset('car.n.01').lemmna_names()
wn.synset('car.n.01').definition()
wn.synset('carn.n.01').examples()

	## ambiguous word : 'car'
wn.synsets('car')
for synset in wn.synsets('car'):
			print(synset.lemmna_names())
wn.lemmas('car')
```

** hypernym : 상위어 ↔ hyponym : 하위어

** holonym : 전체어 ↔ meronym : 부분어

```python
motorcar = wn.synset('car.n.01')

hypo = motorcar.hyponyms()
len(hypo)

hyper = motorcar.hypernyms()
len(hyper)

mero = wn.synset('car.n.01').part_meronyms()
len(mero)

holo = wn.synset('car.n.01').part_holonyms()
len(holo)

```

** stem word

```python
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
lemmatizer.lemmatize('cooking')
lemmatizer.lemmatize('cooking', pos='v')
lemmatizer.lemmatize('leaves', pos='n')
```
