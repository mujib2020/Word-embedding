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
2) GloVe(by Stanford)
3) FastText (by facebook): It is an extension of Word2Vec.
4) ELMo (Embeddings from Language Models): It uses a deep, bi-directional LSTM
5) BERT (Bidirectional Encoder Representations from Transformers): BERT learns word representations by jointly conditioning on both left and right context in all layers using the Transformer architecture.
6) GPT (Generative Pretrained Transformer): Similar to BERT but uses a left-to-right context and is trained to predict the next word in a sentence, which results in word embeddings as a byproduct of its language model.
7) Transformer-based Embeddings: Recent language models like GPT-3, T5, and others use transformer networks to produce context-aware embeddings that reflect the different meanings a word can have in different contexts.


![image](https://github.com/mujib2020/Word-embedding/assets/61886262/7b80b1f4-a467-4b8d-85d1-28f8f11e13fc)

Source: Youtube - codebasics

Sources: 
1) https://www.tensorflow.org/text/guide/word_embeddings
2) Book: Python Natural Language Processing Cookbook - by Zhenya Antic

 ## pretrained word2vec model
 we will use a pretrained word2vec model
####  install gensim package to load and use the model
!pip install gensim
