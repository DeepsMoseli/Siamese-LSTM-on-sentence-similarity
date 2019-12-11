# quora-question-pairs
detection of question repeated questions

__This is a sesond attempt at the Quora questions kaggle challange i worked on a few years back using classical features.__<br>
In this iteration I first attempt to use word2vec embeddings, then bert embedings, and finally training embeddings with the model.<br>
The final model implemented is Siamese LSTM to classify pairs of sentences as either the same question or different.<br>

## Preprocessing
* Removed special characters
* Transform contractions into full form
* Limit text to 16 words
* padding for text length less than 16

## Embeddings
* Tried Word2Vec
* Trained embedding together with model (with zero masking) 

## Model architecture
* Input1 -> LSTM1(128)
* Input2 -> LSTM2(128)
* subtract[LSTM1, LSTM2]
* Dense(128, ReLu) -> Dense(64, ReLu) -> Dense(1, Sigmoid)

## Results
* __40 epochs__
* loss: 0.5062 - acc: 0.7699
* val_loss: 0.389 - val_acc: 0.823
* improvements will be made, tune parameters

* for inference check __GetScore.py__
* Included Trained LSTM in rep, link to Word2vec embedding
