from pathlib import Path
import os
import pandas as pd
import matplotlib.pyplot as plt

files_path = 'C:\\Users\\rajee\\Documents\\Thesis_code\\processed_data'

# Get mean of picture_rt column per sentence-label pair across all csvs also save condition for each pair to make scatterplot based on condition
data_pairs = {}
for file in os.listdir(files_path):
    if file.endswith('.csv'):
        df_path = os.path.join(files_path, file)
        df = pd.read_csv(df_path)
        for element, label, rt, condition in zip(df['words'], df['final_word_NL'], df['picture_rt'], df['condition']):
            try:
                key = (element, label)
                if key not in data_pairs:
                    data_pairs[key] = {'rts': [], 'condition': condition}
                data_pairs[key]['rts'].append(rt)
            except Exception as e:
                print(f'Error processing entry with sentence {element} and target word {label}: {e}')
                continue
        
# Calculate mean rt for each pair
mean_data_pairs = {}
for key, value in data_pairs.items():
    mean_rt = sum(value['rts']) / len(value['rts'])
    mean_data_pairs[key] = {'mean_rt': mean_rt, 'condition': value['condition']}

# make scatterplot per condition
conditions = ['congruent', 'incongruent', 'neutral']
plt.figure(figsize=(10,6))
for condition in conditions:
    x = []
    y = []
    for key, value in mean_data_pairs.items():
        if value['condition'] == condition:
            x.append(condition)
            y.append(value['mean_rt'])
    plt.scatter(x, y, label=condition)
plt.xlabel('Condition')
plt.ylabel('Mean Picture RT')
plt.title('Mean Picture RT per Condition')
plt.legend()
plt.savefig('plots/mean_rt_participants.png')


        