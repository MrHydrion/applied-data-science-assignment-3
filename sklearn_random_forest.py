import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

# snowball stemmer is a text-processin tool to reduce words to their base or root form
stemmer = SnowballStemmer('english')

# load the csv's as pandas
df_train = pd.read_csv('train.csv/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('test.csv/test.csv', encoding="ISO-8859-1")
# df_attr = pd.read_csv('../input/attributes.csv')
df_pro_desc = pd.read_csv('product_descriptions.csv/product_descriptions.csv')

# gets the number of rows in training df
num_train = df_train.shape[0]

# return a string into its base stemmer form
def str_stemmer(s):
	return " ".join([stemmer.stem(word) for word in s.lower().split()])

# counts how many times words of str1 come for in str2
def str_common_word(str1, str2):
	return sum(int(str2.find(word)>=0) for word in str1.split())

# combines both datasets (axis = row) and resets index
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
# merges on id
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

# changing values with its base term
df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))

# getting the lengh of the search term (amount of words in the search term)
df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']
 
# gives how many words of the search terms appear in the product title
# gives how many words of the search terms appear in the product description
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

# drop unnecessary columns
df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)

# split df back to train and test
# extract the ids
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]
id_test = df_test['id']

# create training and testing
y_train = df_train['relevance'].values
X_train = df_train.drop(['id','relevance'],axis=1).values
X_test = df_test.drop(['id','relevance'],axis=1).values

#RFR is an ensemble method that makes decisions tree (number of trees, how far tree can grow, fix for randomness)
#BR training multiple RFR on a subset of the data (number of models, amount of sample each models has, fix for randomness)
rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# make a pd dataframe as a csv
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)

# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
# from nltk.stem.snowball import SnowballStemmer

# stemmer = SnowballStemmer('english')

# df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
# df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
# # df_attr = pd.read_csv('../input/attributes.csv')
# df_pro_desc = pd.read_csv('../input/product_descriptions.csv')

# num_train = df_train.shape[0]

# def str_stemmer(s):
# 	return " ".join([stemmer.stem(word) for word in s.lower().split()])

# def str_common_word(str1, str2):
# 	return sum(int(str2.find(word)>=0) for word in str1.split())


# df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)

# df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

# df_all['search_term'] = df_all['search_term'].map(lambda x:str_stemmer(x))
# df_all['product_title'] = df_all['product_title'].map(lambda x:str_stemmer(x))
# df_all['product_description'] = df_all['product_description'].map(lambda x:str_stemmer(x))

# df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)

# df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title']+"\t"+df_all['product_description']

# df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
# df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))

# df_all = df_all.drop(['search_term','product_title','product_description','product_info'],axis=1)

# df_train = df_all.iloc[:num_train]
# df_test = df_all.iloc[num_train:]
# id_test = df_test['id']

# y_train = df_train['relevance'].values
# X_train = df_train.drop(['id','relevance'],axis=1).values
# X_test = df_test.drop(['id','relevance'],axis=1).values

# rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
# clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission.csv',index=False)

