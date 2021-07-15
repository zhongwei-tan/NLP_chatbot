import pandas as pd
import numpy as np
import gensim
import jieba

qa_df = pd.read_csv("data/qa_data.csv")
questions = qa_df["question"].values
answers = df["answer"].values

model_path = "~/NLP_data/wiki.model"
model = gensim.models.Word2Vec.load(model_path)

def sen2vec(sentence):
    segment = jieba.lcut(sentence)
    vec = np.zeros(100)

    for s in segment:
        try:
            vec += model.wv[s]
        except KeyError:
            pass
    
    vec /= len(segment)
    return vec