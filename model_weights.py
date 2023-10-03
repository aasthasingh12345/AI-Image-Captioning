#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
import re
import nltk
from nltk.corpus import stopwords
import string
import json
from time import time
import pickle
from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input, decode_predictions
#from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.models import Model, load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import keras.utils as image
from keras.layers import Input,Dense,Dropout,Embedding,LSTM
from tensorflow.keras.layers import concatenate,add
import pickle


# In[54]:


model=load_model("model_weights/best_model.h5")


# In[57]:


model_temp=load_model("model_weights/best_model.h5")


# In[58]:


model_resnet = Model(model_temp.input,model_temp.layers[-2].output)


# In[59]:


def preprocess_img(img):
  img = image.load_img(img,target_size=(224,224))
  img = image.img_to_array(img)
  img = np.expand_dims(img,axis=0)
  # Normalization
  img = preprocess_input(img)
  return img


# In[60]:


IMG_PATH="Images/"
img = preprocess_img(IMG_PATH+"1000268201_693b08cb0e.jpg")
plt.imshow(img[0])
plt.axis("off")
plt.show()


# In[61]:


def encode_image(img):
  img = preprocess_img(img)
  feature_vector = model_renet.predict(img)
  feature_vector = feature_vector.reshape(1,feature_vector.shape[1])
  #print(feature_vector.shape)
  return feature_vector


# In[62]:


import json
descriptions=None
with open("descriptions_1.txt",'r') as f:
    descriptions=f.read()
json_acceptable_string=descriptions.replace("'","\"")
descriptions=json.loads(json_acceptable_string)


# In[63]:


print(type(descriptions))


# In[76]:


enc=encode_image("667626_18933d713e")


# In[ ]:





# In[67]:


#Vocab
vocab =set()
for key in descriptions.keys():
    [vocab.update(sentence.split()) for sentence in descriptions[key]]
print("Vocab Size :%d"% len(vocab))


# In[68]:


#Total number of words acress all the sentences
total_words=[]
for key in descriptions.keys():
    [total_words.append(i) for des in descriptions[key] for i in des.split()]

print("Total Words %d"%len(total_words))


# In[ ]:





# In[ ]:





# In[11]:


len(total_words)


# In[22]:


word_to_idx={}
idx_to_word={}

for i,word in enumerate(total_words):
    word_to_idx[word]=i+1
    idx_to_word[i]=word


# In[26]:


with open("word_to_idx.pkl","wb") as w2i:
    pickle.dump(word_to_idx,w2i)
with open("idx_to_word.pkl","wb") as i2w:
    pickle.dump(idx_to_word,i2w)


# In[36]:


with open("./storage/word_to_idx.pkl",'rb') as w2i:
    word_to_idx=pickle.load(w2i)
with open("./storage/idx_to_word.pkl",'rb') as i2w:
    idx_to_word=pickle.load(i2w)


# In[40]:


word_to_idx


# In[41]:


idx_to_word


# In[72]:


def predict_captions(photo):
 in_text="startseq"
 for i in range(max_len):
  sequence=[word_to_idx[w] for w in in_text.split() if w in word_to_idx]
  sequence=pad_sequences([sequence],max_len,padding='post')
        
  ypred=model.predict([photo,sequence])
  ypred=ypred.argmax()  #word with max prob always-Greedy Sample
  word=idx_to_word[ypred]
  in_text +=(' '+word)
        
  if word=='endseq':
    break
            
  final_caption=in_text.split()[1:-1]
  final_caption=' '.join(final_caption)
        
  return final_caption


# In[73]:


def caption_this_image(image):
    enc=encode_image(image)
    caption=predict_caption(enc)
    return caption


# In[ ]:




