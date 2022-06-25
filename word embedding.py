#pip install genism
import gensim
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
# train model
embedding = Word2Vec(sentences, min_count=1,window=5,size=32,sg=0)
embedding['more']
#embedding.most_similar('the', topn=1))
embedding.most_similar('sentence', topn=5)