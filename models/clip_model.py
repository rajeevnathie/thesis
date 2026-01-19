import torch
import clip
import os
import numpy as np
import matplotlib.ticker as ticker
from PIL import Image
from sentence_transformers import  SentenceTransformer, util
import matplotlib.pyplot as plt

device = "cpu"
img_model_name = "ViT-B/32"
# img_model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"

image_model, preprocess = clip.load(img_model_name, device=device)

text_model_name = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
text_model = SentenceTransformer(text_model_name, device="cpu")
#text_model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v1", device="cpu")
#text_model = SentenceTransformer("sentence-transformers/stsb-xlm-r-multilingual")
#text_model = SentenceTransformer("Thaweewat/gte-multilingual-base-m2v-512")


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

image_folder = "C:\\Users\\rajee\\Documents\\Thesis_code\\images\\images"
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))]


#text_inputs = clip.tokenize(labels).to(device)


text_embeddings = text_model.encode(labels, convert_to_tensor=True, normalize_embeddings=True, device=device)
#text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

image_embeddings_list = []
for img in image_files:
    img_path = os.path.join(image_folder, img)
    image = Image.open(img_path).convert('RGB')
    image_in = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embeddings = image_model.encode_image(image_in)
        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
    
    image_embeddings_list.append(image_embeddings)

image_embeddings = torch.cat(image_embeddings_list, dim=0)

# image_embeddings = text_model.encode(image_files, convert_to_tensor=True, normalize_embeddings=True, device=device)

similarity_matrix = util.cos_sim(text_embeddings, image_embeddings)

pic_sim_matrix = util.cos_sim(image_embeddings, image_embeddings)
word_sim_matrix = util.cos_sim(text_embeddings, text_embeddings)

# # Convert to numpy then save so I can see whole file
sim_matrix_np = similarity_matrix.cpu().numpy()
pic_sim_np = pic_sim_matrix.cpu().numpy()
word_sim_np = word_sim_matrix.cpu().numpy()

row_max_indices = np.argmax(sim_matrix_np, axis=1)


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
plt.savefig(f'plots/rdm_60x60_img_{img_model_name.replace("/","-")}.png')
plt.show()

plt.figure(figsize=(8,8))
im = plt.imshow(word_sim_np, cmap='viridis', interpolation='nearest')
cbar = plt.colorbar(label='Correlation')
cbar.locator = ticker.LinearLocator(numticks=5) 
cbar.update_ticks()
plt.title("RDM 60x60 word-word")
plt.xlabel(f'Labels ({text_model_name})')
plt.ylabel(f'Labels ({text_model_name})')
plt.savefig(f'plots/rdm_60x60_txt_{text_model_name.replace("/","-")}.png')
plt.show()