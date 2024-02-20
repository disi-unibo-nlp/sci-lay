import os
import umap
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel


method = "t-sne"
split = "train"

def get_embeddings():

    # Replace 'model_name' with the name or path of the desired Hugging Face model
    model_name = "roberta-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    dataset = load_dataset("paniniDot/sci_lay")[split]
    documents = [doc for doc in dataset["plain_text"]]

    # Define batch size
    batch_size = 2
    all_cls_embeddings = []
    # Process sentences in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(documents), batch_size)):
            batch = documents[i:i+batch_size]

            # Tokenize the batch of sentences
            tokenized_batch = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
            tokenized_batch = {key: value.to(device) for key, value in tokenized_batch.items()}


            # Forward pass through the model
            outputs = model(**tokenized_batch)

            # Extract the CLS embedding from the last layer
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            
            # Append the cls_embedding values to the list
            all_cls_embeddings.append(cls_embedding.cpu().detach())

    # Concatenate the list of tensors into a single tensor
    all_cls_embeddings = torch.cat(all_cls_embeddings, dim=0)

    # Save the list of cls_embeddings to a file
    torch.save(all_cls_embeddings, f'cls_embeddings_{split}.pt')


def main():
    if not os.path.exists(f'cls_embeddings_{split}.pt'):
        get_embeddings()
    
    all_cls_embeddings = torch.load(f'cls_embeddings_{split}.pt')
    dataset = load_dataset("paniniDot/sci_lay")[split]
    journals = np.asarray([journal for journal in dataset["journal"]])

    # Select up to 100 examples for each journal
    selected_points = []
    for journal in np.unique(journals):
        indices = np.where(journals == journal)[0] # [:100]
        selected_points.extend(indices)
    
    all_cls_embeddings = all_cls_embeddings[selected_points, :]
    journals = [journals[i] for i in selected_points]

    if method == "t-sne":
        # Convert the tensor to a NumPy array
        embedding_array = all_cls_embeddings.numpy()

        # Use t-SNE to reduce dimensions to 2
        tsne = TSNE(n_components=2, random_state=42, verbose=2)
        reduced_embeddings = tsne.fit_transform(embedding_array)
    else:
        # Use UMAP to reduce dimensionality from 1024 to 2
        reducer = umap.UMAP(n_components=2)
        reduced_embeddings = reducer.fit_transform(all_cls_embeddings)

    fig = px.scatter(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], color=journals)
    fig.write_image(f"{method}_plot_{split}.png", width=1920, height=1080)

if __name__ == '__main__':
    main()