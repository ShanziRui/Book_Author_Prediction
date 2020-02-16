from IPython.display import display
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
import os
import random
import load
import glob
import re
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
import nltk
# nltk.download()
from nltk.corpus import stopwords
nltk.download('stopwords')


txt_files = glob.glob("*.txt")


books = []
for txt in txt_files:
    books.append(load.Book(txt, False,
                           False, os.getcwd()))

print("*"*12+" Top 7 most frequently downloaded books "+"*"*12)
for book in books:
    print(book.title, book.author, sep=': ')
print()


stemmer = SnowballStemmer('english')
words = stopwords.words('english')
df = pd.DataFrame([], columns=['Title', 'Author',
                               'NumChapters', 'Contents', 'Clean', 'Tokens', 'NumDoc', 'Documents'])

# population_size = {}
author = []
size = []

print("*"*12+" Population size of each book "+"*"*12)
for book in books:
    index = 1
    # print(len(book.chapters))
    total = 0
    for chapter in book.chapters:
        Title = book.title
        Author = book.author
        NumChapters = index
        Contents = " ".join(chapter)
        Clean = " ".join([stemmer.stem(i) for i in re.sub(
            "[^a-zA-Z]", " ", Contents).split() if i not in words]).lower()
        Tokens = Clean.split(' ')
        NumDoc = len(Tokens)//150
        if NumDoc == 0:
            Documents = []
        else:
            Documents = np.array_split(Tokens, NumDoc)
        total = total + NumDoc
        df2 = pd.DataFrame([[Title, Author, NumChapters, Contents, Clean, Tokens, NumDoc, Documents]], columns=[
                           'Title', 'Author', 'NumChapters', 'Contents', 'Clean', 'Tokens', 'NumDoc', 'Documents'])
        df = df.append(df2, ignore_index=True)
        index = index + 1
    print(book.title, total)
    author.append(book.author)
    size.append(total)

print()

# visualization of population size
population = pd.Series(size,
                       index=author)
df = pd.DataFrame({'population': population})
df = df.sort_values(by='population')

my_range = list(range(1, len(df.index)+1))

fig, ax = plt.subplots(figsize=(5, 3.5))

plt.hlines(y=my_range, xmin=0,
           xmax=df['population'], color='#007ACC', alpha=0.2, linewidth=5)

plt.plot(df['population'], my_range, "o",
         markersize=5, color='#007ACC', alpha=0.6)

ax.set_xlabel('population', fontsize=15, fontweight='black', color='#333F4B')
ax.set_ylabel('')

ax.tick_params(axis='both', which='major', labelsize=12)
plt.yticks(my_range, df.index)

fig.text(-0.23, 0.96, 'Population Size', fontsize=15,
         fontweight='black', color='#333F4B')

ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
ax.spines['bottom'].set_smart_bounds(True)

ax.spines['bottom'].set_position(('axes', -0.04))
ax.spines['left'].set_position(('axes', 0.015))

plt.savefig('population_size.png', dpi=300, bbox_inches='tight')


# print(df)


training = pd.DataFrame([], columns=['Author', 'Title', 'Document'])


for book in books:
    Title = book.title
    Author = book.author
    bk = df[df['Author'] == Author]
    bk = bk[['Author', 'Title', 'Documents']]
    Documents = []
    for index, row in bk.iterrows():
        for docs in row['Documents']:
            doc = ' '.join(docs)
            Documents.append(doc)
    Samples = random.sample(Documents, 200)
    for Document in Samples:
        df3 = pd.DataFrame([[Author, Title, Document]], columns=[
            'Author', 'Title', 'Document'])
        training = training.append(df3, ignore_index=True)


# print(training)


# Visualization of number of words in each document distribution
sns.set()

fig, ax = plt.subplots(figsize=(15, 6))

ax.set_title("Distribution of number of words", fontsize=16)
ax.set_xlabel("Number of words")
document_lengths = np.array(list(map(len, training.Document.str.split(' '))))
sns.distplot(document_lengths, bins=50, ax=ax)
plt.savefig('no_words_distribution.png', dpi=300, bbox_inches='tight')


print("*"*12+" 1400 documents after sampling "+"*"*12)
print()
print(training)
print()


print("*"*12+" Sampling size of each book "+"*"*12)
print(training.groupby('Author').count())
print()

# Feature Engineering
# Bag of Word
vectorizer_bow = CountVectorizer(ngram_range=(1, 2))
text_counts_bow = vectorizer_bow.fit_transform(training.Document)
feature_names_bow = vectorizer_bow.get_feature_names()

# Visualization of word frequency of training data
sum_words = text_counts_bow.sum(axis=0)
words_freq = [(word, sum_words[0, idx])
              for word, idx in vectorizer_bow.vocabulary_.items()]
words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
print(words_freq[:10])

word = []
count = []
for (w, c) in words_freq:
    word.append(str(w))
    count.append(int(c))
word_count = pd.Series(count, index=word)
df = pd.DataFrame({'count': word_count})

sns.set()

nr_top_words = 50
fig, ax = plt.subplots(1, 1, figsize=(20, 5))
sns.barplot(list(range(nr_top_words)),
            df['count'].values[:nr_top_words], palette='hls', ax=ax)
ax.set(ylim=(0, 1500))
ax.set_xticks(list(range(nr_top_words)))
ax.set_xticklabels(df.index[:nr_top_words], fontsize=14, rotation=90)
ax.set_title("Word Frequencies", fontsize=16)
plt.savefig('word_frequency.png', dpi=300, bbox_inches='tight')


# TF-IDF
vectorizer_tf = TfidfVectorizer(ngram_range=(1, 2))
text_counts_tf = vectorizer_tf.fit_transform(training['Document'])
feature_names_tf = vectorizer_tf.get_feature_names()

# LDA
no_topics = 5
lda = LatentDirichletAllocation(
    n_components=no_topics, learning_method='online', learning_offset=50., random_state=0)
lda_X = lda.fit_transform(text_counts_bow)

# LDA V2
# sc = StandardScaler()
# lda = LatentDirichletAllocation(n_components=1)
# lda_X = lda.fit_transform(X_train, y_train)
# X_test = lda.transform(X_test)


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

# no_top_words = 10
# display_topics(lda, feature_names_bow, no_top_words)


# Split train and test set 10-fold validation
def cross_val(algorithm, X, y):
    split = 10
    skf = StratifiedKFold(n_splits=split)
    n_fold = 1
    accuracy_ls = []

    if algorithm == "svm":
        print("*"*12+" Result for Support Vector Machine "+"*"*12)
    elif algorithm == "knn":
        print("*"*12+" Result for K-Nearest Neighbor "+"*"*12)
    elif algorithm == "mlp":
        print("*"*12+" Result for MLP "+"*"*12)
    elif algorithm == "dt":
        print("*"*12+" Result for Decision Tree "+"*"*12)

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if algorithm == "svm":
            report, accuracy = pred_svm(X_train, X_test, y_train, y_test)
        elif algorithm == "knn":
            report, accuracy = pred_knn(X_train, X_test, y_train, y_test)
        elif algorithm == "mlp":
            report, accuracy = pred_mlp(X_train, X_test, y_train, y_test)
        elif algorithm == "dt":
            report, accuracy = pred_dt(X_train, X_test, y_train, y_test)
        accuracy_ls.append(accuracy)
        print("The accuracy for Fold "+str(n_fold)+" : "+str(accuracy))
        print(report)
        print()
        n_fold += 1

    print("The Average Accuracy: "+str(np.mean(accuracy_ls)))
    print('\n')
    return accuracy_ls


# support vector machine
def pred_svm(X_train, X_test, y_train, y_test):
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return report, accuracy


# K nearest neighbor
def pred_knn(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(20)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return report, accuracy

# MLP classifier


def pred_mlp(X_train, X_test, y_train, y_test):
    mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)
    mlp.fit(X_train, y_train.values.ravel())
    y_pred = mlp.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    return report, accuracy


# Decision Tree
def pred_dt(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier(max_depth=6).fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    report = classification_report(y_test, y_pred)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    plt.suptitle("Decision surface of a decision tree")
    plt.legend(loc='lower left', borderpad=5, handletextpad=5)
    plt.axis()
    plt.figure(figsize=(19, 7), dpi=150)
    plot_tree(dt, filled=True, fontsize=5)
    plt.axis('off')
    plt.show()
    return report, accuracy


# modeling
# BOW
X = text_counts_bow
y = training['Author']

# TF-IDF
# X = text_counts_tf
# y = training['Author']

# LDA
# X = lda_X
# y = training['Author']

# Modeling
acc_svm = cross_val("svm", X, y)
acc_knn = cross_val("knn", X, y)
acc_mlp = cross_val("mlp", X, y)
acc_dt = cross_val("dt", X, y)


# Visualization of cross-validation with BOW as feature
model = ['SVM']*10 + ['KNN']*10 + ['MLP']*10 + ['DT'] * 10
accuracy = acc_svm + acc_knn + acc_mlp + acc_dt

sns.set()
cv_df = pd.DataFrame({'model_name': model, 'accuracy': accuracy})
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.savefig('bow_cross_validation.png', dpi=300, bbox_inches='tight')


# Visualization of confusion matrix with MLP model
mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=1000)

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
    text_counts_bow.toarray(), training.Author, training.index, test_size=0.3, random_state=0)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=training.Author.drop_duplicates(), yticklabels=training.Author.drop_duplicates())
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')


# Error analysis for MPL model
author_label = training.Author.drop_duplicates()
index_id = author_label.factorize()[0]
author_label = author_label.values

for predicted in index_id:
    for actual in index_id:
        if predicted != actual and conf_mat[actual, predicted] >= 10:
            print("'{}' predicted as '{}' : {} examples.".format(
                author_label[actual], author_label[predicted], conf_mat[actual, predicted]))
            display(training.loc[indices_test[(y_test == author_label[actual]) & (y_pred == author_label[actual])]][[
                    'Author', 'Document']])
            print('')
