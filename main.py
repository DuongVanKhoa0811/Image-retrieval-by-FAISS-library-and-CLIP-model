import argparse
import numpy as np
import streamlit as st
from PIL import Image
import os
import clip
import faiss
import torch
import time


@st.cache_resource        
def get_model():
    model, preprocess = clip.load('ViT-B/32', device=device)
    model.eval()
    return model, preprocess

@st.cache_resource        
def encode_image_clip():
    filenames = os.listdir('./val2017') ###########
    filenames.sort()
    if not os.path.exists('Embedded_Files'):
        os.mkdir('Embedded_Files')
    else:
        return filenames
    chunk_size = 100
    group_filenames = [filenames[i:i+chunk_size] for i in range(0, len(filenames), chunk_size)]
    print('Start the encoding process!!!')
    for batch_id, batch_filenames in enumerate(group_filenames):
        print('Encode ' + (str)(batch_id + 1) + '/' + (str)(len(group_filenames)))
        images = []
        for filename in batch_filenames:
            image = preprocess(Image.open('./val2017/' + filename)).unsqueeze(0).to(device)
            images.append(image)
        images = torch.stack(images, dim=0)
        images = images.squeeze()
        encode_images = model.encode_image(images).detach().numpy()
        torch.save(encode_images, 'Embedded_Files/encode_batch_' + (str)(batch_id) + '.pt')
        print('Save encode images batch', batch_id + 1)
    print('Finished the encoding process!!!')
        
    return filenames

@st.cache_resource        
def indexing(file_index=None):
    if file_index is not None:
        print('Load file index!!!')
        index = faiss.read_index(file_index)
        print('Finished load file index!!!')
    else:
        encode_files = os.listdir('Embedded_Files')
        print('Start the index process!!!')
        index = faiss.IndexFlatIP(len_clip_feature)
        for encode_id in range(len(encode_files)):
            encode_file = 'encode_batch_' + (str)(encode_id) + '.pt'
            print(encode_file, 'Index ' + (str)(encode_id + 1) + '/' + (str)(len(encode_files)))
            encode_images = torch.load('Embedded_Files/' + encode_file)
            index.add(encode_images)
        faiss.write_index(index, 'database.index')
        print('Finished the index process and save it as database.index for later use!!!')
    return index

global device
global args
global len_clip_feature 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
len_clip_feature = 512
# get model
model, preprocess = get_model()

# get image and encode it using the aforementioned model
filenames = encode_image_clip()

# index the image using the faiss library
index = indexing('database.index') # 


# query the images base on the query
def get_images(query, index, model, image_names, top_results=20):
    start_time = time.time()
    content = clip.tokenize([query]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(content).cpu().numpy()
    dists, idxs = index.search(text_features, top_results)
    end_time = time.time()

    results = {}
    for i in range(top_results):
        result = {'image': Image.open('./val2017/' + image_names[idxs[0][i]]), 'distance': (float)(dists[0][i])}
        results[i] = result
    return results, (end_time - start_time)


def streamlit_app(model, index, image_names):
    st.title('Multimodal Search with CLIP and FAISS')
    st.subheader('Student ID: 19125099')
    query = st.text_input('Input query here')
    
    # start query the image follow the content in the text box
    if query or st.button('Search'):
        results, query_time = get_images(query, index, model, image_names)
        st.write(f'Processing time: {query_time:.3f}s')
        colums = st.columns(3)
        count_colums = 0
        for idx in range(len(results)):
            result = results[idx]
            image = result['image']
            colums[count_colums].image(image, caption='Image top ' + str(idx + 1), use_column_width=True)
            if count_colums >= 2:
                count_colums = 0
            else:
                count_colums += 1

    # view all images, which are used for the query
    if st.button('View all images'):
        colums = st.columns(3)
        count_colums = 0
        st.write('100 image for demo purpose!!!')
        for image_name in image_names[:100]:
            image = Image.open('./val2017/' + image_name)
            colums[count_colums].image(image, use_column_width=True)
            if count_colums >= 2:
                count_colums = 0
            else:
                count_colums += 1


if __name__ == '__main__':
    streamlit_app(model, index, filenames)