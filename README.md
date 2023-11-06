# Word-embedding
Word embedings are mathematical mrepresentations of texts. In other words, a word embedding is a representation of a word in a real-valued vector. The inputs in the ML models are numbers. When working with text, the first thing we must do is come up with a strategy to convert strings to numbers (or to "vectorize" the text) before feeding it to the model. Three strategies for doing so: 
1) One-hot encodings
2) Encode each word with a unique number
3) Word embeddings

*Word embeddings* are preferred because they capture semantic similarities between words in lower-dimensional space, unlike one-hot encodings which are sparse and don't convey meaning, or unique numbers which are arbitrary and non-semantic.



Sources: 
1) https://www.tensorflow.org/text/guide/word_embeddings
2) Book: Python Natural Language Processing Cookbook - by Zhenya Antic

 ## pretrained word2vec model
 we will use a pretrained word2vec model
####  install gensim package to load and use the model
!pip install gensim
