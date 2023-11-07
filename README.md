# Word-embedding
Word embedings are mathematical mrepresentations of texts. In other words, a word embedding is a representation of a word in a real-valued vector. The inputs in the ML models are numbers. When working with text, the first thing we must do is come up with a strategy to convert strings to numbers (or to "vectorize" the text) before feeding it to the model. Some strategies for doing so: 
1) One-hot encodings
2) Encode each word with a unique number
3) Bag of words
4) TF-IDF
5) Word embeddings

**Word embeddings** are preferred because they capture semantic similarities between words in lower-dimensional space, unlike one-hot encodings which are sparse and don't convey meaning, or unique numbers which are arbitrary and non-semantic.

**Some word embedding methods**
There are several methods to generate word embeddings, with varying complexity and linguistic sophistication. Some of the most prominent methods include:
1) Word2Vec (by Google):  There are two architectures
   
   Skip-gram: Predicts context words given a target word.
   Continuous Bag of Words (CBOW): Predicts the target word from a set of context words.
   
3) GloVe(by Stanford)
4) FastText (by facebook): It is an extension of Word2Vec.
5) ELMo (Embeddings from Language Models): It uses a deep, bi-directional LSTM
6) BERT (Bidirectional Encoder Representations from Transformers): BERT learns word representations by jointly conditioning on both left and right context in all layers using the Transformer architecture.
7) GPT (Generative Pretrained Transformer): Similar to BERT but uses a left-to-right context and is trained to predict the next word in a sentence, which results in word embeddings as a byproduct of its language model.
8) Transformer-based Embeddings: Recent language models like GPT-3, T5, and others use transformer networks to produce context-aware embeddings that reflect the different meanings a word can have in different contexts.

Sources: 
1) https://www.tensorflow.org/text/guide/word_embeddings
2) Gensim: https://radimrehurek.com/gensim/models/word2vec.html
3) Book: Python Natural Language Processing Cookbook - by Zhenya Antic

 ## Pretrained and train on a custom dataset
 At first we will use a pretrained word2vec model and then train word2vec model on a custom dataset. Here is the link for the code: [Word-embedding.ipynb](https://github.com/mujib2020/Word-embedding/blob/master/Word-embedding.ipynb)https://github.com/mujib2020/Word-embedding/blob/master/Word-embedding.ipynb

