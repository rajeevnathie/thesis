from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageFile
from pathlib import Path
from multilingual_clip import pt_multilingual_clip
from transformers import AutoTokenizer
from elmoformanylangs import Embedder
from transformers import AutoTokenizer, AutoModelForMaskedLM
import clip
import os
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import spacy

device = 'cpu'

# Load image and text models
img_model, preprocess = clip.load("ViT-B/32", device='cpu')
model_name = 'M-CLIP/XLM-Roberta-Large-Vit-B-32'
text_model = pt_multilingual_clip.MultilingualCLIP.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load roberta base transformer
model_name = 'xlm-roberta-large'
transformer_tokenizer = AutoTokenizer.from_pretrained(model_name)
transformer_model = AutoModelForMaskedLM.from_pretrained(model_name)
transformer_model.eval()
mask_token = transformer_tokenizer.mask_token


# Load ELMo model
elmo_model_path = "C:/Users/rajee/Documents/Thesis_code/elmo_model/164"
e = Embedder(elmo_model_path)

nlp = spacy.load("nl_core_news_sm", disable=["parser", "ner"])

# Change path here to path where your image folder is located
images_path = "C:\\Users\\rajee\\Documents\\Thesis_code\\images\\images"

# Pairs of data that need to be changed due to typos in original dataset
change_data_pairs = {'Vanochttend hoorde ze de': 'Vanochtend hoorde ze de', 'Bij dat gebied stroomt een': 'Door dat gebied stroomt een',
                     'In spanje schijnt vaak de': 'In Spanje schijnt vaak de', 'De man leest elke avond de': 'De man leest elke ochtend de'}

# Function to calculate surprisal using transformer
def get_next_word_surprisal(sentence_prefix, target_word):
    """
    Calculates surprisal of 'target_word' being the word 
    after 'sentence_prefix'.
    """
    masked_input = f"{sentence_prefix} {mask_token}"

    inputs = transformer_tokenizer(masked_input, return_tensors="pt")
    
    mask_indices = (inputs.input_ids == transformer_tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    mask_index = mask_indices[0]

    with torch.no_grad():
        outputs = transformer_model(**inputs)
        logits = outputs.logits[0, mask_index, :]
    
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    target_ids = transformer_tokenizer.encode(target_word, add_special_tokens=False)
    
    if not target_ids:
        return None
        
    first_token = target_ids[0]
    
    target_probe = probs[first_token].item()
    surprisal = -np.log(target_probe + 1e-20)
    
    return surprisal

def get_transformer_embeddings(texts):
    """
    Extract embeddings from transformer model's hidden states.
    """
    tokenized = transformer_tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = transformer_model(**tokenized, output_hidden_states=True)
    # Mean pooling of last layer hidden states
    embeddings = outputs.hidden_states[-1].mean(dim=1)
    return embeddings

def extract_data_from_df(df, data_pairs):
    for element, pic_name, label, condition in zip(df['words'], df['picture'], df['final_word_NL'], df['condition']):
        for f in os.listdir(images_path):
            if pic_name == f.split('_', 1)[1]:
                pic_path = os.path.join(images_path, f)
                break
        if pic_path is None:    
            print(f'Image for picture name {pic_name} not found. Skipping entry with sentence {sentence} and label {label}.')
            continue
        try:
            data_element = {'picture': pic_path, 'condition': condition}
        except Exception as e:
            print(f'Error processing entry with sentence {element} and target word {label}: {e}')
            continue
        if element in change_data_pairs:
            element = change_data_pairs[element]
        data_pairs[(element, label)] = data_element 
    return data_pairs

# Encode images using clip image encoder
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

# Encode texts using clip text encoder
def encode_texts(texts):
    with torch.no_grad():
        text_embeddings = text_model.forward(texts, tokenizer)
    return text_embeddings

def encode_elmo_texts(texts):
    tokenized_sentences = []
    for s in texts:
        doc = nlp(s)
        tokens = [token.text for token in doc]
        tokenized_sentences.append(tokens) 
    return tokenized_sentences

data_pairs = {}
forward_dim = 512


# Get all csvs from data folder
for file in os.listdir('C:\\Users\\rajee\\Documents\\Thesis_code\\data'):
    print(f'Processing file: {file}')
    try:
        # exclude file for with sub009 in its title since data is not usable
        if file.endswith('.csv') and 'sub009' not in file:
            df_path = os.path.join('C:\\Users\\rajee\\Documents\\Thesis_code\\data', file)
            df = pd.read_csv(df_path)
        else:
            print(f'Skipping file: {file}')
            continue
    except FileNotFoundError:
        print(f'File {file} not found. Skipping.')
    if not data_pairs:
        try:
            data_pairs = extract_data_from_df(df, data_pairs)
        except Exception as e:
            print(f'Error extracting data from {file}: {e}')
        # get unique images, sentences, labels in set and convert to sorted list to be able to retrieve indexes
        images = sorted(list({row.get('picture') for row in data_pairs.values()}))
        sentences = sorted(list({keys[0] for keys in data_pairs.keys()}))
        labels = sorted(list({keys[1] for keys in data_pairs.keys()}))
        # Assert length of 120 can use any number here based on dataset
        assert len(images) == 60, f'Expected 60 images, got {len(images)}'
        assert len(sentences) == 120, f'Expected 120 sentences, got {len(sentences)}'
        assert len(labels) == 60, f'Expected 60 labels, got {len(labels)}'
        print(f'Unique images: {len(images)}, Unique sentences: {len(sentences)}, Unique labels: {len(labels)}')
        img_embeddings = encode_image_paths(images)
        #encode texts
        clip_text_embeddings = encode_texts(sentences)
        clip_label_embeddings = encode_texts(labels)
        #encode texts and labels using transformer
        transformer_text_embeddings = get_transformer_embeddings(sentences)
        transformer_label_embeddings = get_transformer_embeddings(labels)
        #Elmo encode texts
        tokenized_sentences = encode_elmo_texts(sentences)
        tokenized_labels = encode_elmo_texts(labels)
        elmo_sentence_embeddings = e.sents2elmo(tokenized_sentences, output_layer=-2)
        elmo_label_embeddings = e.sents2elmo(tokenized_labels, output_layer=-2)
    # Compute cosine distances and add to dataframe
    for i, (sentence, img, label, condition) in enumerate(zip( df['words'], df['picture'], df['final_word_NL'], df['condition'])):
        if sentence in change_data_pairs:
            sentence = change_data_pairs[sentence]
            df.at[i, 'words'] = sentence
        image = data_pairs[(sentence, label)]['picture'].split('\\')[-1].split('_', 1)[1]
        if data_pairs[(sentence, label)]['condition']!= condition: 
            print(f'Data mismatch for sentence {sentence} and target word {label} because of mismatching condition. Skipping.')
            continue
        if image != img:
            print(f'Image mismatch for sentence {sentence} and target word {label} because of wrong target picture. Skipping.')
            continue
        # Find index of sentence, image, label in original sets
        sentence_index = list(sentences).index(sentence)
        image_index = list(images).index(data_pairs[(sentence, label)]['picture'])
        label_index = list(labels).index(label)
        # Get cosine distances
        cos_dis_text_clip = util.cos_sim(clip_text_embeddings[sentence_index], img_embeddings[image_index])
        cos_dis_transformer = util.cos_sim(transformer_text_embeddings[sentence_index], transformer_label_embeddings[label_index])
        cos_dis_elmo_nonclip = util.cos_sim(torch.tensor(elmo_sentence_embeddings[sentence_index][2, -1, :forward_dim]), elmo_label_embeddings[label_index][2, -1, :forward_dim])
        df.at[i, 'cos_dis_text_clip'] = cos_dis_text_clip.item()
        df.at[i, 'cos_dis_transformer'] = cos_dis_transformer.item()
        df.at[i, 'cos_dis_elmo_nonclip'] = cos_dis_elmo_nonclip.item()
        if data_pairs[(sentence, label)].get('cos_dis_text_clip') is None:
            data_pairs[(sentence, label)]['cos_dis_text_clip'] = cos_dis_text_clip.item()
        if data_pairs[(sentence, label)].get('cos_dis_transformer') is None:
            data_pairs[(sentence, label)]['cos_dis_transformer'] = cos_dis_transformer.item()
        if data_pairs[(sentence, label)].get('cos_dis_elmo_nonclip') is None:
            data_pairs[(sentence, label)]['cos_dis_elmo_nonclip'] = cos_dis_elmo_nonclip.item()

        # calculate all cosine similarities between current clip text embedding and all clip image embeddings also for elmo and non clip
        clip_similarities = util.cos_sim(clip_text_embeddings[sentence_index], img_embeddings)
        elmo_sentence_emb = torch.tensor(elmo_sentence_embeddings[sentence_index][-1, :forward_dim]).unsqueeze(0)
        elmo_label_embs = torch.stack([torch.tensor(emb[-1, :forward_dim]) for emb in elmo_label_embeddings])
        if len(elmo_sentence_emb.shape) == 3:
            elmo_sentence_emb = elmo_sentence_emb.mean(dim=1) 
        if len(elmo_label_embs.shape) == 3:
            elmo_label_embs = elmo_label_embs.squeeze(1)
        elmo_similarities = util.cos_sim(elmo_sentence_emb, elmo_label_embs)

        # scale the logits by a temperature parameter (e.g., 0.01)
        temperature = 0.01
        elmo_scaled_logits = elmo_similarities / temperature
        #calculate surprisal
        elmo_probabilities = torch.softmax(elmo_scaled_logits, dim=1)
        transformer_surprisal = get_next_word_surprisal(sentence, label)
        elmo_surprisal = -torch.log(elmo_probabilities[0, label_index] + 1e-10)
        df.at[i, 'transformer_surprisal'] = transformer_surprisal.item()
        df.at[i, 'elmo_surprisal'] = elmo_surprisal.item()

    # Save updated dataframe to new csv
    new_csv = df.to_csv(f'C:\\Users\\rajee\\Documents\\Thesis_code\\processed_data\\{file.replace(".csv", "_llm.csv")}', index=False)
    print(f'Processed and saved updated {file}')

# Create scatter plot for each model based on condition with on y axis the average cosine distance and on x axis the condition in different colors
# model_types = [('cos_dis_text_clip', 'CLIP'), ('cos_dis_transformer', 'Text-encoder'), ('cos_dis_elmo_nonclip', 'ELMo')]
# for model_key, model_name in model_types:
#     conditions = {'congruent': [], 'incongruent': [], 'neutral': []}
#     for (sentence, label), values in data_pairs.items():
#         condition = values['condition']
#         avg_cos_dis = values[model_key]
#         conditions[condition].append(avg_cos_dis)
    
#     plt.figure(figsize=(10, 6))  
#     for condition, cos_dis_list in conditions.items():
#         #scatter plot all values of list
#         plt.scatter([condition]*len(cos_dis_list), cos_dis_list, label=condition)
    
#     plt.title(f'Average Cosine Similarity by Condition - {model_name}')
#     plt.xlabel('Condition')
#     plt.ylabel('Average Cosine Similarity')
#     plt.legend()
#     plt.savefig(f'C:\\Users\\rajee\\Documents\\Thesis_code\\plots\\cos_dis_{model_name}.png')
#     plt.close()