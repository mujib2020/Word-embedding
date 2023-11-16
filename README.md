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


 ## Word2Vec: Pretrained and train on a custom dataset
 At first we will use a pretrained word2vec model and then train word2vec model on a custom dataset. Here is the implementation (code): Word-embedding.ipynb

### Drawback of Word2Vec:

Let's consider these sentences:

1. He was not fair in his role.
2. There will be a fun fair in the city of Dallas.
3. 
Word2Vec may generate the same vector for the word "fair" in both sentences. However, the word 'fair' has different meanings in these contexts.

**Remedy with BERT:** To address this issue, we can use BERT, which generates contextualized embeddings. Unlike Word2Vec, BERT considers the entire context of a word within a sentence, allowing it to distinguish between different usages of the same word.
 ## BERT:
BERT generates sophisticated word embeddings that capture the context of a word within a sentence. Unlike traditional models, BERT examines the full context of a word by looking at the words that come before and after it, resulting in rich, nuanced representations that vary depending on the word's usage.

BERT implementation: See the code above:  Word_embedding_BERT.ipynb

**Sources**

*Word2vec:*
1) https://www.tensorflow.org/text/guide/word_embeddings
2) Gensim: https://radimrehurek.com/gensim/models/word2vec.html
3) Book: Python Natural Language Processing Cookbook - by Zhenya Antic

*BERT*:
1. BERT models: https://tfhub.dev/google/collections/bert/1
2. Text preprocessing layer: https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3
3. https://jalammar.github.io/illustrated-bert/
4.  https://www.techtarget.com/searchenterpriseai/definition/BERT-language-model#:~:text=BERT%2C%20which%20stands%20for%20Bidirectional,calculated%20based%20upon%20their%20connection.
