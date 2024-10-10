"""
Paraphrase Mining:

Paraphrase mining is the task of finding paraphrases (texts with identical / similar meaning) in a large corpus of sentences. 
In Semantic Textual Similarity we saw a simplified version of finding paraphrases in a list of sentences. 
The approach presented there used a brute-force approach to score and rank all pairs.
"""
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import paraphrase_mining

model = SentenceTransformer("all-MiniLM-L6-v2")

# Single list of sentences - Possible tens of thousands of sentences
sentences = [
    "The cat sits outside",
    "A man is playing guitar",
    "I love pasta",
    "The new movie is awesome",
    "The cat plays in the garden",
    "A woman watches TV",
    "The new movie is so great",
    "Do you like pizza?",
]

paraphrases = paraphrase_mining(model, sentences)

for paraphrase in paraphrases[0:10]:
    score, i, j = paraphrase
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))
    
    
"""
The output is:
The new movie is awesome                 The new movie is so great               Score: 0.8939
The cat sits outside             The cat plays in the garden             Score: 0.6788
I love pasta             Do you like pizza?              Score: 0.5096
I love pasta             The new movie is so great               Score: 0.2560
I love pasta             The new movie is awesome                Score: 0.2440
A man is playing guitar                  The cat plays in the garden             Score: 0.2105
The new movie is awesome                 Do you like pizza?              Score: 0.1969
The new movie is so great                Do you like pizza?              Score: 0.1692
The cat sits outside             A woman watches TV              Score: 0.1310
The cat plays in the garden              Do you like pizza?              Score: 0.0900
"""    