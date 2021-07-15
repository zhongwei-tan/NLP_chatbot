from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import multiprocessing

input_file = "~/NLP_data/wiki.txt"
output_file = "~/NLP_data/wiki.model"

print("Training Word2Vec...")
model = Word2Vec(
    LineSentence(input_file),
    vector_size=100,
    window=5,
    sg=1,
    hs=0,
    negative=5,
    min_count=5,
    workers=multiprocessing.cpu_count(),
)

print(f"Training completed and model save at {output_file}")
model.save(output_file)