import pandas as pd

df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.head(3)

df.loc[:500].to_csv('movie_data.csv')
df = pd.read_csv('movie_data.csv', nrows=5000)
print('SMALL DATA SUBSET CREATED FOR TESTING')


from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer(stop_words='english',
                        max_df=0.3,
                        max_features=5000)
X = count.fit_transform(df['review'].values)


from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch')
X_topics = lda.fit_transform(X)

lda.components_.shape

n_top_words = 5
feature_names = count.get_feature_names()

for topic_idx, topic in enumerate(lda.components_):
    print("Topic %d:" % (topic_idx + 1))
    print(" ".join([feature_names[i]
                    for i in topic.argsort()\
                        [:-n_top_words - 1:-1]]))



horror = X_topics[:, 2].argsort()


print("fafdds")
count = 0
for iter_idx, movie_idx in enumerate(horror):
    count += 1
    print('\nHorror movie #%d:' % (iter_idx + 1))
    #print(df['review'][movie_idx][:300], '...')
print(count)
