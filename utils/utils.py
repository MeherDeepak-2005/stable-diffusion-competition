import pandas as pd
from PIL import Image
import torch


def gpt2model_test(gpt2_model, clip_model, preprocess_image, emb_gen):
    df = pd.read_csv('../dataset/prompts.csv')
    submissions = pd.read_csv("../dataset/sample_submission.csv")

    img_paths = df['imgId'].to_list()

    actual_embeddings = submissions['val']
    actual_embeddings = actual_embeddings.to_list()
    actual_embeddings = torch.Tensor(actual_embeddings)

    predicted_embeddings = []

    for image in img_paths:
        image = Image.open("./dataset/images/" + image + '.png')
        image = preprocess_image(image).unsqueeze(0)

        with torch.no_grad():
            image_features = clip_model(image)
            image_features = image_features.view(-1, 32, 32)

            image_embeddings = emb_gen(image_features.to('cuda'))

            predicated_embedding = gpt2_model(image_embeddings)

            predicted_embeddings.append(predicated_embedding)

    x = torch.cat(predicted_embeddings).flatten()

    similarity = torch.nn.functional.cosine_similarity(x.cpu(), actual_embeddings, dim=0) \
        .mean().item()

    return similarity


def single_model_test(model, preprocess_image):
    df = pd.read_csv('./dataset/prompts.csv')
    submissions = pd.read_csv("./dataset/sample_submission.csv")

    img_paths = df['imgId'].to_list()

    actual_embeddings = submissions['val']
    actual_embeddings = actual_embeddings.to_list()
    actual_embeddings = torch.Tensor(actual_embeddings)

    predicted_embeddings = []

    for image in img_paths:
        image = Image.open("./dataset/images/" + image + '.png')
        image = preprocess_image(image).unsqueeze(0).to(model.device)

        with torch.no_grad():
            predicated_embedding = model(image)

            predicted_embeddings.append(predicated_embedding)

    x = torch.cat(predicted_embeddings).flatten()

    similarity = torch.nn.functional.cosine_similarity(x.cpu(), actual_embeddings, dim=0) \
        .mean()

    if torch.isnan(similarity).item() is True:
        print(x)

    return similarity


def generate_embeddings():
    from sentence_transformers import SentenceTransformer
    import numpy as np

    dataframe = pd.read_csv("../datasets/train_dataset/eval.csv")
    prompts = dataframe['Prompt'].to_list()

    st = SentenceTransformer(model_name_or_path="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = st.encode(prompts, show_progress_bar=True)
    np.savez('../datasets/train_dataset/eval-embeddings.npz', embeddings=embeddings)


# def temporary(element: str):
#     element = element.split('/')
#     element.pop(1)
#     element.insert(1, 'datasets')
#     element = '/'.join(element)
#
#     return element
#
#
# df = pd.read_csv('../datasets/train_dataset/train.csv')
# df['image_path'] = df['image_path'].apply(temporary)
# df.to_csv('../datasets/train_dataset/train.csv')
#
# df = pd.read_csv('../datasets/train_dataset/eval.csv')
# df['image_path'] = df['image_path'].apply(temporary)
# df.to_csv('../datasets/train_dataset/eval.csv')
