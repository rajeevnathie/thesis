import spacy
import torch
import numpy as np
from elmoformanylangs import Embedder

print(f"GPU: {torch.cuda.get_device_name(0)}")

# Change model path to where the model is loaded, first download the model
model_path = "C:/Users/rajee/Documents/Thesis_code/elmo_model/164"
e = Embedder(model_path)
print("Elmo model loaded")

nlp = spacy.load("nl_core_news_sm", disable=["parser", "ner"])

sentences = ["Dit is een voorbeeldzin", "Dit is een voorbeeldzin met extra tekst.",
             "De kat zit op de mat", "De kat zit op de mat en kijkt naar buiten.",
             "Het weer is vandaag mooi" ,"Het weer is vandaag mooi en zonnig."]
    
tokenized_sentences = []
for s in sentences:
    doc = nlp(s)
    tokens = [token.text for token in doc]
    tokenized_sentences.append(tokens)

full_embeddings = e.sents2elmo(tokenized_sentences, output_layer=-2)
forward_dim = 512

# Get resutls of embeddings of sentences after second forward layer
for i in range(0, len(full_embeddings), 2):
    len1 = len(tokenized_sentences[i])
    embed_1 = full_embeddings[i][2,len1-1,:forward_dim]
    embed_2 = full_embeddings[i+1][2,len1-1,:forward_dim]

    if not np.allclose(embed_1, embed_2, atol=1e-6):
        print(f"Embeddings differ after second forward layer for sentences {i+1} and {i+2}.")
        print(full_embeddings[i][2,len1,forward_dim])
        print(full_embeddings[i+1][2,len1,forward_dim])
    else:
        print(f"Embeddings are same for {i+1} and {i+2} up until word with index {len1-1}.")