import numpy as np
import warnings
import sys
from tqdm import tqdm
warnings.filterwarnings("ignore")

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize as wt, sent_tokenize

import keras
from keras import backend as k
from keras.models import load_model


#I ddnt need to create a tf session for running on my machine u will add if u need it here


model1 = Word2Vec.load("word2vec_256.model")
siamese_model = load_model('my_model_siamese_Lstm_final.h5')

def inference(x1,x2):
    #tokenize and pad
    x1=wt(x1.lower().strip())
    x2=wt(x2.lower().strip())
    
    if len(x1)>=16:
        x1=x1[:16]
    else:
        while(len(x1)<16):
            x1.append("pad")
            
    if len(x2)>=16:
        x2=x2[:16]
    else:
        while(len(x2)<16):
            x2.append("pad")
    q1=[]
    q2=[]
    for word in x1:
        try:
            q1.append(model1.wv.word_vec(word))
        except Exception as e:
            q1.append(model1.wv.word_vec("pad"))
            continue
    for word2 in x2:
        try:
            q2.append(model1.wv.word_vec(word2))
        except Exception as e2:
            q2.append(model1.wv.word_vec("pad"))
            continue
    
    x1 = np.asarray(q1,dtype='float32').reshape((1,16,256))
    x2 = np.asarray(q2,dtype='float32').reshape((1,16,256))
    sim_prob = siamese_model.predict([x1,x2])
    return sim_prob[0][0]


def score_all(tts,script_lists):
    # this function takes tts as a string
    # and script_lists as a long list of the script strings
    hold = ""
    counter=0
    sent=[]
    for word in script_lists:
        if counter==16:
            sent.append(hold)
            hold=""
            counter=0
            print(1)
        else:
            counter+=1
            hold = hold + word +" "
            next
    if hold!="":
        sent.append(hold)
    result = [inference(tts,ref_) for ref_ in sent]
    return max(result)



script = """Note that if we were to run the t-SNE again with different parameters, 
                we may observe some similarities to this result, but we’re not 
                guaranteed to see the exact same patterns. t-SNE is not deterministic.
                Relatedly, tightness of clusters and distances between clusters are not
                always meaningful. It is meant primarily as an exploratory tool, rather
                than as a decisive indicator of similarity.Why are Eastern European
                countries like Poland and Hungary so reluctant to allow immigrants
                from the Middle East into their territories such as france. We can see at a glance that 
                WSJ has both the highest standard deviation and the largest range, with
                the lowest minimum sentiment compared to any other top source. I dont know
                about Hinton’s views on the matter, but he is sort of Father of Neural Nets, 
                so he surely will have some ground article titles. To verify this 
                rigorously would require a hypothesis test,if they speak french, which is beyond the scope of 
                this post, but it’s an interesting potential finding and future direction.
                For right-wing libertarians: Why should a factory owner receive more profit
                than the workers who constructed and maintain said factory?""" 


text_to_speech = ["Does France really have a poor work ethic or is it just myth",
                  "How can entrepreneurs get business loans when they have limited revenue?",
                  "to see the exact same patterns. t-SNE is not deterministic. Relatedly",
                  "tightness of clusters and distances between clusters are not in the class",
                  "What is the most inappropriate experience you’ve had with a student?",
                  "Why is Geoffrey Hinton suspicious of backpropagation and wants AI to start over",
                  "I’ll also engineer a new feature",
                  "What are some examples of Lee Kuan Yew standing up to bigger powers like the US",
                  "japan or China? ghana or morocco? germany or englang?",
                  "Who is the president of lesotho today",
                  "Is it a common thing to like food when stressed?",
                  "In america the biggest economy or not in 2019?",
                  "I just want the semester to end,lol im so tired i cant even sleep anymore"]


scores = []
pbar = tqdm(range(len(text_to_speech)))
for x1_input in (text_to_speech):
    scores.append(score_all(x1_input,wt(script))) #wt is tokenizer of script, wont be called if the scripts is list of word strings
    pbar.update(1)
pbar.close()

print("similarity probability for each phrase from text to speech against entire script: \n%s"%scores)
print("Overall score: %s"%round(np.mean(scores)*100,1)+"% sense similarity")

len(wt(script))
217/16
