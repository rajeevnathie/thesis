from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageFile
from pathlib import Path
from multilingual_clip import pt_multilingual_clip
from transformers import AutoTokenizer
import clip
import os
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt


device = 'cpu'

img_model, preprocess = clip.load("ViT-B/32", device='cpu')
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'
text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

df = pd.read_csv('C:\\Users\\rajee\\Documents\\Thesis_code\\data\\results_sub001_20250507_1031.csv')


images_path = "C:\\Users\\rajee\\Documents\\Thesis_code\\images\\images"
conditions = ['congruent', 'incongruent', 'neutral']

congruent_sentences = []
congruent_images = []

incongruent_sentences = []
incongruent_images = []

neutral_sentences = []
neutral_images = []

for element, pic_name, condition in zip(df['words'], df['picture'], df['condition']):
    for f in os.listdir(images_path):
        if pic_name == f.split('_', 1)[1]:
            pic_path = os.path.join(images_path, f)
    if condition == conditions[0]:
        congruent_sentences.append(element)
        congruent_images.append(pic_path)
    elif condition == conditions[1]:
            incongruent_sentences.append(element)
            incongruent_images.append(pic_path)
    elif condition == conditions[2]:
            neutral_sentences.append(element)
            neutral_images.append(pic_path)


def encode_image_paths(image_files):
    image_embeddings_list = []
    for img in image_files:
        image = Image.open(img).convert('RGB')
        image_in = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embeddings = img_model.encode_image(image_in)
            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
        
        image_embeddings_list.append(image_embeddings)

    img_embeddings = torch.cat(image_embeddings_list, dim=0)
    return img_embeddings


def encode_texts(texts):
    with torch.no_grad():
        text_embeddings = text_model.forward(texts, tokenizer)
    return text_embeddings

def plot_rdm(sim_matrix, row_max_indices, title_suffix=""):
    plt.figure(figsize=(8,8))
    plt.imshow(sim_matrix, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Correlation')
    for i, j in enumerate(row_max_indices):
        plt.scatter(j, i, marker='o', s=40, edgecolor='white', facecolor='black')
    plt.title("RDM 60x60 " + title_suffix)
    plt.xlabel('Images (ViT-B/32)')
    plt.ylabel('Sentences (M-CLIP/XLM-Roberta-Large-Vit-B-32)')
    plt.show()

congruent_image_embeddings = encode_image_paths(congruent_images)
incongruent_image_embeddings = encode_image_paths(incongruent_images)
neutral_image_embeddings = encode_image_paths(neutral_images)

congruent_text_embeddings = encode_texts(congruent_sentences)
incongruent_text_embeddings = encode_texts(incongruent_sentences)
neutral_text_embeddings = encode_texts(neutral_sentences)

cos_sim_congruent = util.cos_sim(congruent_text_embeddings, congruent_image_embeddings)
cos_sim_incongruent = util.cos_sim(incongruent_text_embeddings, incongruent_image_embeddings)
cos_sim_neutral = util.cos_sim(neutral_text_embeddings, neutral_image_embeddings)

sim_matrix_congruent = cos_sim_congruent.cpu().numpy()
sim_matrix_incongruent = cos_sim_incongruent.cpu().numpy()
sim_matrix_neutral = cos_sim_neutral.cpu().numpy()

row_max_congruent = np.argmax(sim_matrix_congruent, axis=1)
row_max_incongruent = np.argmax(sim_matrix_incongruent, axis=1)
row_max_neutral = np.argmax(sim_matrix_neutral, axis=1)

plot_rdm(sim_matrix_congruent, row_max_congruent, title_suffix="(Congruent)")
plot_rdm(sim_matrix_incongruent, row_max_incongruent, title_suffix="(Incongruent)")
plot_rdm(sim_matrix_neutral, row_max_neutral, title_suffix="(Neutral)")