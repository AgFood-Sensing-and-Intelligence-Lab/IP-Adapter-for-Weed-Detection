import torch
from PIL import Image
import open_clip
import numpy as np
import os
from tqdm import tqdm
import time

import pandas as pd

# from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns
import traceback

def list_dir(path, list_name, extension, return_names=False):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            list_dir(file_path, list_name, extension)
        else:
            if file_path.endswith(extension):
                if return_names:
                    list_name.append(file)
                else:
                    list_name.append(file_path)
    try:
        list_name = sorted(list_name)
    except Exception as e:
        print(e)
    return list_name


particular_cats = [
    'Lambsquarters',
    'PalmerAmaranth',
    'Waterhemp',
    'MorningGlory',
    'Purslane',
    'Goosegrass',
    'Carpetweed',
    'SpottedSpurge',
    'Ragweed',
    'Eclipta',
]


def generate_embedding_and_save_transformer(im_path, model, save_dir, preprocess=None, use_BioCLIP=False):
    image = Image.open(im_path)
    if use_BioCLIP:
        pass
    else:
        image = image.convert("RGB")
    if preprocess:
        # image = np.array(image).astype(np.float32) / 255.0
        # image = torch.from_numpy(image)
        image = preprocess(image).unsqueeze(0)
    im_name = os.path.basename(im_path)
    with torch.no_grad(), torch.cuda.amp.autocast():
        with torch.no_grad():
            if use_BioCLIP:
                image_features = model.encode_image(image)
            else:
                # image_features = model(image, output_hidden_states=True).hidden_states[-2]
                output_CLIPVisionModel = model(image, output_hidden_states=False)
                image_features = output_CLIPVisionModel.image_embeds
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features_np = image_features.detach().numpy().squeeze()
        save_path = os.path.join(save_dir, f'{im_name}_embedding.npy')
        np.save(save_path, image_features_np)


def generate_embedding_and_save_openclip(im_path, model, save_dir, preprocess=None):
    image = Image.open(im_path)
    if preprocess:
        image = preprocess(image).unsqueeze(0)
    im_name = os.path.basename(im_path)
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)

        # text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        # text_features /= text_features.norm(dim=-1, keepdim=True)

        image_features_np = image_features.detach().numpy()
        save_path = os.path.join(save_dir, f'{im_name}_embedding.npy')
        np.save(save_path, image_features_np)

    #     text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    # print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]


def generate_embeddings_using_transformer():
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection
    from torchvision import transforms

    use_BioCLIP=True
    if use_BioCLIP:
        image_encoder_path = r'StableDiffusionWeedV2\bioclip\model'
        image_encoder_path = r'StableDiffusionWeedV2\bioclip\model\open_clip_pytorch_model.bin'
        image_encoder, preprocess_train, preprocess_val= open_clip.create_model_and_transforms('ViT-B-16', pretrained=image_encoder_path)
        image_encoder.eval()  
    else:
        image_encoder_path = r'D:\Models\clip_vision\clip_image_encoder'
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(image_encoder_path)
        image_encoder.requires_grad_(False)

    model = image_encoder

    size = 224
    if use_BioCLIP:
        # for BioVLIP
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
        ])
        preprocess = preprocess_val
    else:
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        # clip_image_processor = CLIPImageProcessor()
        preprocess = transform
    for particular_cat in particular_cats:
        im_dir = rf'D:\Dataset\WeedData\weed_10_species\train2017_real_object_in_box\{particular_cat}'
        
        save_dir = r'D:\Dataset\WeedData\weed_10_species\train2017_real_object_in_box\Embedding_BioCLIP'
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        im_paths = list_dir(im_dir, [], '.jpg')
        import random
        random.seed(0)
        random.shuffle(im_paths)
        # text = tokenizer(["a weed", "a plant", "nothing"])
        cnt = 0
        for im_path in tqdm(im_paths):
            if cnt >= 200:
                break
            try:
                generate_embedding_and_save_transformer(im_path, model, save_dir, preprocess, use_BioCLIP)
                cnt += 1
            except Exception as e:
                print(e)
                continue


def generate_embeddings_BioCLIP_using_openclip():
    # To train BIOCLIP, we initialize from OpenAI’s CLIP
    # weights [69] with a ViT-B/16 vision transformer
    image_encoder_path = r'StableDiffusionWeedV2\bioclip\model\open_clip_pytorch_model.bin'

    # pretrained also accepts local paths
    # model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained=image_encoder_path)
    model.eval()  # model in train mode by default, impacts some models with BatchNorm or stochastic depth active
    tokenizer = open_clip.get_tokenizer('ViT-B-16')
    for particular_cat in particular_cats:
        im_dir = rf'D:\Dataset\WeedData\weed_10_species\train2017_real_object_in_box\{particular_cat}'
        save_dir = r'D:\Dataset\WeedData\weed_10_species\train2017_real_object_in_box\BioCLIP_Embedding'

        im_paths = list_dir(im_dir, [], '.jpg')
        import random
        random.seed(0)
        random.shuffle(im_paths)
        # text = tokenizer(["a weed", "a plant", "nothing"])
        cnt = 0
        for im_path in tqdm(im_paths):
            if cnt >= 200:
                break
            try:
                generate_embedding_and_save_openclip(im_path, model, save_dir, preprocess)
                cnt += 1
            except Exception as e:
                print(traceback.format_exc())
                print(e)
                continue


def generate_t_SNT():
    
    
    src_dir = r'D:\Dataset\WeedData\weed_10_species\train2017_real_object_in_box\Embedding_BioCLIP_from_openclip'

    embedding_paths = list_dir(src_dir, [], '.npy')
    embeddings = []
    ys = []
    for x in embedding_paths:
        name = os.path.basename(x)
        sku = name.split('_')[0]
        embedding = np.load(x)
        embeddings.append(embedding)
        ys += [sku]
    embeddings_0 = np.array(embeddings)
    ys_0 = np.array(ys)
    visualize_embedding(embeddings_0, ys_0)


def visualize_embedding(embeddings, ys):
    X = embeddings.squeeze()
    y = ys
    print(X.shape, y.shape)

    feat_cols = ['pixel'+str(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feat_cols)
    df['y'] = y
    df['label'] = df['y'].apply(lambda i: str(i))

    X, y = None, None

    print('Size of the dataframe: {}'.format(df.shape))

    rndperm = np.random.permutation(df.shape[0])

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df[feat_cols].values)

    df['pca-one'] = pca_result[:, 0]
    df['pca-two'] = pca_result[:, 1]
    df['pca-three'] = pca_result[:, 2]

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df.loc[rndperm, :],
        legend="full",
        alpha=0.3
    )

    # df_subset = df.loc[rndperm[:N],:].copy()
    df_subset = df.loc[rndperm, :].copy()
    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

    data_subset = df_subset[feat_cols].values
    tsne_results = tsne.fit_transform(data_subset)

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    df_subset['tsne-2d-one'] = tsne_results[:, 0]
    df_subset['tsne-2d-two'] = tsne_results[:, 1]

    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", 10),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    plt.show()


def main():
    try:
        # generate_embeddings_BioCLIP_using_openclip()
        # generate_embeddings_using_transformer()
        generate_t_SNT()
        pass
    except Exception as e:
        print(e)
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
