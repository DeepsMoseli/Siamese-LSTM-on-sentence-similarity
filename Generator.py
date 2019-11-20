# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:40:52 2019
generator function to use model.fit_generator
Majority of threadsafe_iter code adopted from: 
https://github.com/Cheng-Lin-Li/SegCaps/blob/master/utils/threadsafe.py
@author: Deeps
"""
import threading
import numpy as np

class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe"""
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g

@threadsafe_generator
def get_samples(x,y,batch_size,mode="train"):
    """One X generator"""
    sizeX = len(x)
    while True:
        try:
            index = [np.random.randint(0,sizeX) for k in range(batch_size)]
            examples=x[index]
            labels=y[index]
            yield (examples,labels)
        except Exception as e:
            print(e)
            break

@threadsafe_generator
def get_samples_2_inputs(x1,x2,y,batch_size,mode="train"):
    """Use this call in the case of Siamese networks with 2 X inputs"""
    sizeX = len(x1)
    while True:
        try:
            index = [np.random.randint(0,sizeX) for k in range(batch_size)]
            examples1=x1[index]
            examples2=x2[index]
            labels=y[index]
            yield [[examples1,examples2],labels]
        except Exception as e:
            print(e)
            break

"""USAGE: 
train_gen = get_samples(sentence1_list,,binary_similarity_class,32)
train_gen = get_samples_2_inputs(sentence1_list,sentence2_list,binary_similarity_class,32)
"""