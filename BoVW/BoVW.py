import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PIL import Image
import cv2

import joblib
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import vq


class BagOfVisualWords:
    def __init__(self, root_dir='/kaggle/input/the-hyper-kvasir-dataset/labeled_images',
                    method='sift',
                ):
        self.root_dir = root_dir
        self.df = pd.read_csv(f'{root_dir}/image-labels.csv')
        self.labels = tuple(self.df['Finding'].unique())
        
        if method =='sift':
            self.extractor = cv2.SIFT_create()
        elif method == 'orb':
            self.extractor = cv2.ORB_create()
        elif method == 'surf':
            self.extractor = cv2.xfeatures2d.SURF_create()
        else:
            raise ValueError(f'Unsupported feature detection method: {method}')
        
        # helper 
        self.tfidf = 1
        
        
    def extract_descriptors(self, sample_size=1000):
        '''Extract descriptors from sample_size images
        :param method: method to extract descriptors e.g. ORB, SIFT, SURF, etc
        :param sample_size: size of sample. (We likely use a small sample in real-world scenario, 
            where whole dataset is big)
            
        :return: all descriptors of sample_size images
        :rtype: list
        '''
        np.random.seed(0) # reproducibility
        sample_idx = np.random.randint(0, len(self.df)+1, sample_size).tolist()
        assert len(sample_idx) == 1000, 'Invalid sampling'

        descriptors_sample_all = [] # each image has many descriptors, descriptors_sample_all
        # is all descriptors of sample_size images
        
        # loop each image > extract > append
        for n in sample_idx:
            # descriptors extracting
            img_descriptors = self._feature_detecting(n)
            if img_descriptors is not None:
                for descriptor in img_descriptors:
                    descriptors_sample_all.append(np.array(descriptor))
                    
        # convert to single numpy array
        descriptors_sample_all = np.stack(descriptors_sample_all)
        
        return descriptors_sample_all

    def build_codebook(self, 
                            all_descriptors: np.array,
                            k: int = 200,
                            iters=1,
                            codebook_path='bovw-codebook.pkl',
                        ):
        '''Building visual vocabulary (visual words)
        :param all_descriptors: array of descriptors
        :param k: #cluster (centroids)
        :param codebook_path: path to saving codebook
        
        :return: #centroids, codebook

        '''
        codebook, variance = kmeans(all_descriptors, k, iters)
        
        return codebook
    
    def get_embedding(self, idx: int,
                        k: int,
                        codebook: np.array,
                    ):
        '''Get embeddings of image[idx] (image > descriptors > project in codebook > frequencies vectors)
        :param idx: image index
        
        :return: frequencies vector (can consider as embedding)
        '''
        img_descriptors = self._feature_detecting(idx)
        img_visual_words, distance = vq(img_descriptors, codebook)
        # create a frequency vector
        img_frequency_vector = np.zeros(k) # k = #cluster = #word in codebook
        for word in img_visual_words:
            img_frequency_vector[word] += 1

        return img_frequency_vector * self.tfidf
    
    def _tf_idf(self, sample_embeddings: np.array):
        '''Reweight important features in codebook'''
        
        N = len(self.df)
        df = np.sum(sample_embeddings > 0, axis=0)
        idf = np.log(N/ df)
        tfidf = sample_embeddings * idf
        
        self.tfidf = tfidf
    
    def _feature_detecting(self, idx, grayscale=True):
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
    
    model = BagOfVisualWords(root_dir='/media/mountHDD2/lamluuduc/endoscopy/dataset/hyperKvasir/labeled-images')
    all_descriptors = model.extract_descriptors()
    joblib.dump(all_descriptors, 'all_descriptors.pkl', compress=3) # saving all descriptors
            

        

