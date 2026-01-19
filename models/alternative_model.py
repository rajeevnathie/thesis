import os
import torch
import open_clip
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
from PIL import Image
from torchvision import transforms
from sentence_transformers import SentenceTransformer, util

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

# text_model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

# text_model = SentenceTransformer(text_model_name, device="cpu")


# img_model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
img_model_name = "ViT-H-14"

text_model_name = "laion2b_s32b_b79k"
model, _, preprocess = open_clip.create_model_and_transforms(img_model_name, pretrained=text_model_name)
model = model.to("cpu").eval()


image_folder = "C:\\Users\\rajee\\Documents\\Thesis_code\\images\\images"
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

images = [preprocess(Image.open(f).convert("RGB")) for f in image_files]
images = torch.stack(images, dim=0).to("cpu")

with torch.no_grad():
    image_embeddings = model.encode_image(images)
    image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

tokenizer = open_clip.get_tokenizer(img_model_name)
texts = tokenizer(labels)
with torch.no_grad():
    text_tokens = tokenizer(labels).to("cpu")
    text_embeddings = model.encode_text(text_tokens)
    text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)


similarity_matrix = util.cos_sim(text_embeddings, image_embeddings)

pic_sim_matrix = util.cos_sim(image_embeddings, image_embeddings)
word_sim_matrix = util.cos_sim(text_embeddings, text_embeddings)

# # Convert to numpy then save so I can see whole file
sim_matrix_np = similarity_matrix.cpu().numpy()
pic_sim_np = pic_sim_matrix.cpu().numpy()
word_sim_np = word_sim_matrix.cpu().numpy()

row_max_indices = np.argmax(sim_matrix_np, axis=1)

print("got here")

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
im = plt.imshow(pic_sim_np, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
cbar = plt.colorbar(label='Correlation')
cbar.locator = ticker.LinearLocator(numticks=5) 
cbar.update_ticks()
plt.title("RDM 60x60 picture-picture")
plt.xlabel(f'Images ({img_model_name})')
plt.ylabel(f'Images ({img_model_name})')
plt.savefig(f'plots/rdm_60x60_{img_model_name.replace("/","-")}.png')
plt.show()

plt.figure(figsize=(8,8))
im = plt.imshow(word_sim_np, cmap='viridis', interpolation='nearest')
cbar = plt.colorbar(label='Correlation')
cbar.locator = ticker.LinearLocator(numticks=5) 
cbar.update_ticks()
plt.title("RDM 60x60 word-word")
plt.xlabel(f'Labels ({text_model_name})')
plt.ylabel(f'Labels ({text_model_name})')
plt.savefig(f'plots/rdm_60x60_{text_model_name.replace("/","-")}.png')
plt.show()
