import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
import cv2

import joblib
from sklearn.cluster import KMeans
from scipy.cluster.vq import vq



np.random.seed(0) # reproducibility
class BagOfVisualWords:
    def __init__(self, root_dir: str ='/kaggle/input/the-hyper-kvasir-dataset/labeled_images',
                    method: str ='sift',
                    # k: int = 200,
                    all_descriptors_dir: str =None,
                    codebook_dir: str =None
                ):
        self.root_dir = root_dir
        self.df = pd.read_csv(f'{root_dir}/image-labels.csv')
        self.labels = tuple(self.df['Finding'].unique())
        self.method = method
        
        if method =='sift':
            self.extractor = cv2.SIFT_create()
        elif method == 'orb':
            self.extractor = cv2.ORB_create()
        elif method == 'surf':
            self.extractor = cv2.xfeatures2d.SURF_create()
        else:
            raise ValueError(f'Unsupported feature detection method: {method}')
        
        # helper 
        if codebook_dir is not None:
            self.k, self.codebook = joblib.load(codebook_dir)
            
        if all_descriptors_dir is not None:
            self.all_descriptors = joblib.load(all_descriptors_dir)
            
        self.idf = 1
        self.samples_idx = [] # small sample idx for building visual vocabulary
        
    def extract_descriptors(self, sample_size=1000):
        '''Extract descriptors from sample_size images
        :param method: method to extract descriptors e.g. ORB, SIFT, SURF, etc
        :param sample_size: size of sample. (We likely use a small sample in real-world scenario, 
            where whole dataset is big)
            
        :return: all descriptors of sample_size images
        :rtype: list
        
        # TODO: sample for building visual vocabulary must be balance between classes
        every class include at least one image
        '''
        self.sample_idx = np.random.randint(0, len(self.df)+1, sample_size).tolist()

        descriptors_sample_all = [] # each image has many descriptors, descriptors_sample_all
        # is all descriptors of sample_size images
        
        # loop each image > extract > append
        for n in self.sample_idx:
            # descriptors extracting
            img_descriptors = self._get_descriptors(n)
            if img_descriptors is not None:
                for descriptor in img_descriptors:
                    descriptors_sample_all.append(np.array(descriptor))
                    
        # convert to single numpy array
        descriptors_sample_all = np.stack(descriptors_sample_all)
        
        return descriptors_sample_all

    def build_codebook(self, 
                            all_descriptors: np.array,
                            k: int = 200,
                        ):
        '''Building visual vocabulary (visual words)
        :param all_descriptors: array of descriptors
        :param k: #cluster (centroids)
        :param codebook_path: path to saving codebook
        
        :return: #centroids, codebook

        '''
        kmeans = KMeans(n_clusters=k, random_state=123)
        kmeans.fit(all_descriptors)
        
        return kmeans.cluster_centers_
    
    def get_embedding(self, idx: int,
                        normalized: bool = False,
                        tfidf: bool = False
                    ):
        '''Get embeddings of image[idx] (image > descriptors > project in codebook > frequencies vectors)
        :param idx: image index
        
        :return: frequencies vector (can consider as embedding)
        '''
        img_descriptors = self._get_descriptors(idx)
        img_visual_words, distance = vq(img_descriptors, self.codebook)
        img_frequency_vector = np.histogram(img_visual_words, bins=self.k, density=normalized)[0]
        
        if tfidf:
            self._tf_idf()
            img_frequency_vector = img_frequency_vector * self.idf
            
        return img_frequency_vector
    
    def _tf_idf(self):
        '''TODO: Reweight important features in codebook'''
        
        all_embeddings = []
        for i in range(len(self.df)):
            embedding = self.get_embedding(i)
            all_embeddings.append(embedding)
            
        all_embeddings = np.stack(all_embeddings)
        
        N = len(self.df)
        df = np.sum(all_embeddings > 0, axis=0)
        idf = np.log(N/ df)
        
        return idf
    
    def _get_descriptors(self, idx, grayscale=True):
        '''Extracting descriptors for each image[idx]
        :param method: method to extract features e.g. ORB, SIFT, SURF, etc
        :param idx: image index
        
        :return: descriptors
        :rtype: np.array
        '''
        # get image
        img, _ = self._get_item(idx)
        # preprocessing: convert to grayscale for efficient computing
        if len(img.shape) == 3 and grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # descriptors extracting
        _, img_descriptors = self.extractor.detectAndCompute(img, None)
        
        return img_descriptors
    
    def _get_item(self, idx):
        '''Return pair (image(arr), label)
        :param idx: index of data
        
        :return:
            tuple (image: array, label)
        '''
        # get path of image
        GI_dir = {'Lower GI':'lower-gi-tract',
            'Upper GI':'upper-gi-tract'}

        img = self.df['Video file'][idx]
        gi_tract = GI_dir[self.df['Organ'][idx]]
        classification = self.df['Classification'][idx]
        finding = self.df['Finding'][idx]
        path = f'''{self.root_dir}/{gi_tract}/{classification}/{finding}/{img}.jpg'''
        assert os.path.exists(path) == True, "File does not exist" # dir existance checking

        # read image
        image = np.array(Image.open(path))
        label = self.labels.index(finding)
        
        return image, label
        


if __name__ == '__main__':
    model = BagOfVisualWords(root_dir='/media/mountHDD2/lamluuduc/endoscopy/dataset/hyperKvasir/labeled-images',
                            codebook_dir='bovw_codebook_sift.pkl'
                        )
    # # 1. extracting descriptors
    # all_descriptors = model.extract_descriptors(sample_size=2000)
    # joblib.dump(all_descriptors, f'sample_all_descriptors.pkl', compress=3) # saving all descriptors

    # # 2. building visual vocabulary
    # k = 200
    # all_descriptors = joblib.load('all_descriptors_sift.pkl')
    # codebook = model.build_codebook(all_descriptors, k)
    # joblib.dump((k, codebook), f'bovw_codebook_{model.method}.pkl', compress=3) # saving codebook

    embedding = model.get_embedding(0, normalized=True)
    # plt.bar(list(range(len(embedding))),embedding)
