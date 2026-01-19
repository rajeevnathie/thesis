from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageFile
from pathlib import Path
from multilingual_clip import pt_multilingual_clip
from transformers import AutoTokenizer
import matplotlib.ticker as ticker
import clip
import os
import requests
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

device = 'cpu'

img_model_name = "ViT-B/32" # <-- good model
#img_model_name = "clip-ViT-B-32-multilingual-v1"
#img_model_name = "ViT-H-14"

img_model, preprocess = clip.load(img_model_name, device='cpu') # <-- good model
#img_model = SentenceTransformer(f'sentence-transformers/{img_model_name}', device='cpu') # <-- model 2

text_model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32' # <-- good model
#text_model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
#text_model_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(text_model_name)

tokenizer = AutoTokenizer.from_pretrained(text_model_name)


#text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1', device='cpu')


image_folder = "C:\\Users\\rajee\\Documents\\Thesis_code\\images\\images"
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

# Image encoder 1
image_embeddings_list = []
for img in image_files:
    image = Image.open(img).convert('RGB')
    image_in = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embeddings = img_model.encode_image(image_in)
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    
    image_embeddings_list.append(image_embeddings)

img_embeddings = torch.cat(image_embeddings_list, dim=0)

# Image encoder 2
#img_embeddings = text_model.encode(image_files, convert_to_tensor=True, normalize_embeddings=True, device=device)



labels = ["baby","bij","vlinder","koe","hond","oog","leeuw","lip",
            "muis","haan","staart","bamboe","kerstboom","ei","bos",
            "gras","zon","vuur","boom","water","rivier","banaan","taart",
            "wortel","druiven","slagroom","rugzak","bal","pleister",
            "bezem","kwast","hengel","glas","bril","pistool","hamer",
            "mes","spiegel","pijp","bord","zeep","lepel","handdoek",
            "fluitje","board","bijbel","boek","krant","vliegtuig",
            "brug","kooi","kasteel","kerk","deur","kapstok","iglo",
            "mijter","postzegel","tafel","vaas"
            ]


#text_embeddings = text_model.encode(labels, convert_to_tensor=True, normalize_embeddings=True, device=device)

with torch.no_grad():
    text_embeddings = text_model.forward(labels, tokenizer)

    cos_sim = util.cos_sim(text_embeddings, img_embeddings)
    pic_sim_matrix = util.cos_sim(img_embeddings, img_embeddings)
    word_sim_matrix = util.cos_sim(text_embeddings, text_embeddings)

    sim_matrix_np = cos_sim.cpu().numpy()
    pic_sim_np = pic_sim_matrix.cpu().numpy()
    word_sim_np = word_sim_matrix.cpu().numpy()

    row_max_indices = np.argmax(sim_matrix_np, axis=1)


# print(len(image_files), len(labels))

# for i, (p, lab) in enumerate(zip(image_files, labels)):
#     print(i, os.path.basename(p), "->", lab)

plt.figure(figsize=(8,8))
im = plt.imshow(sim_matrix_np, cmap='viridis', interpolation='nearest')
cbar = plt.colorbar(label='Correlation')
for i, j in enumerate(row_max_indices):
    plt.scatter(j, i, marker='o', s=40, edgecolor='black', facecolor='white')
cbar.locator = ticker.LinearLocator(numticks=5) 
cbar.update_ticks()
plt.title("RDM 60x60")
plt.xlabel(f'Images ({img_model_name})')
plt.ylabel(f'Labels ({text_model_name})')
plt.savefig(f'plots/rdm_60x60_{img_model_name.replace("/","-")}_{text_model_name.replace("/","-")}.png')
plt.show()


plt.figure(figsize=(8,8))
im = plt.imshow(pic_sim_np, cmap='viridis', interpolation='nearest')
cbar = plt.colorbar(label='Correlation')
cbar.locator = ticker.LinearLocator(numticks=5) 
cbar.update_ticks()
plt.title("RDM 60x60 image-image")
plt.xlabel(f'Images ({img_model_name})')
plt.ylabel(f'Images ({img_model_name})')
plt.savefig(f'plots/rdm_60x60_img_{img_model_name.replace("/","-")}.png')
plt.show()

plt.figure(figsize=(8,8))
im = plt.imshow(word_sim_np, cmap='viridis', interpolation='nearest')
cbar = plt.colorbar(label='Correlation')
cbar.locator = ticker.LinearLocator(numticks=5) 
cbar.update_ticks()
plt.title("RDM 60x60 label-label")
plt.xlabel(f'Labels ({text_model_name})')
plt.ylabel(f'Labels ({text_model_name})')
plt.savefig(f'plots/rdm_60x60_txt_{text_model_name.replace("/","-")}.png')
plt.show()