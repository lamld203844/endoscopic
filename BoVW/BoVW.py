from typing import Optional
import os

import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
import cv2

from sklearn.cluster import KMeans, MiniBatchKMeans
from scipy.cluster.vq import vq

class BagOfVisualWords:
    def __init__(
        self,
        root_dir: str = "/kaggle/input/the-hyper-kvasir-dataset/labeled_images",
        descriptors_lake_path: str = None,
        codebook_dir: str = None,
        
        method: str = 'sift',
        extractor_kwargs: dict = None,
    ):
        """Constructor method
        
        :param descriptors_lake_path: str (optional), path to file including all computed descriptors (vectors)
        :param codebook_dir: str (optional), path to visual vocabulary
        
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(f"{root_dir}/image-labels.csv")
        self.labels = tuple(self.df["Finding"].unique())
        
        # In reality in building codebook, choose small sample size idx for efficient 
        self.samples_idx = []  

        # ------ extracting algorithms --------
        self.method = method.lower()
        if self.method == "sift":
            self.extractor = cv2.SIFT_create()
        elif self.method == "orb":
            self.extractor = cv2.ORB_create(**extractor_kwargs)
        elif self.method == "surf":
            self.extractor = cv2.xfeatures2d.SURF_create(**extractor_kwargs)
        else:
            raise ValueError(f"Unsupported feature extracting method: {method}")
            
    def extract_descriptors(self, sample_size: int = 2000,
                                grayscale: bool = True,
                                strongest_percent: float = 1,
                                **extractor_kwargs
                        ) -> np.ndarray:
        """Extract descriptors from sample_size images
        :param method: str, method to extract feature descriptors e.g. ORB, SIFT, SURF, etc
        :param sample_size: size of sample. (We likely use a small sample in real-world scenario,
            where whole dataset is big)
        :param grayscale: bool - if True, convert to gray for efficient computing
        :param strongest_percent: float - get % percent of strongest (based on .response of keypoints)
        descriptors.  

        :return: list, n descriptors x sample_size images

        # TODO: sample for building visual vocabulary must be balance between classes
        every class include at least one image
        """
        # ------- Sampling -----------
        self.sample_idx = np.random.choice(np.arange(0, len(self.df)),
                                            size=sample_size,
                                            replace=False
                                        ).tolist() #  randomly sample sample_size images

        descriptors_sample_all = (
            []
        )  # each image has many descriptors, descriptors_sample_all
        # is a list of all descriptors of sample_size images

        # loop each image > extract > append
        for idx in tqdm(self.sample_idx):
            img_keypoints, img_descriptors = self._get_descriptors_one_img(idx)
            if img_descriptors is not None:
                # filter top_percent strongest keypoint
                sorted_couple = sorted(zip(img_keypoints, img_descriptors), key=lambda x: x[0].response, reverse=True)
                img_keypoints, img_descriptors = zip(*sorted_couple) # unzip
                top = int(len(img_keypoints) * strongest_percent)
                top_descriptors = img_descriptors[:top]               

                for descriptor in top_descriptors:
                    descriptors_sample_all.append(np.array(descriptor))

        # convert to single numpy array
        descriptors_sample_all = np.stack(descriptors_sample_all)

        return descriptors_sample_all

    def build_codebook(
        self,
        descriptors_lake: np.ndarray,
        cluster_algorithm: str = 'kmean',
        k: int = 200,
        batch_size = 1000,
        n_init: int = 10
    ) -> np.ndarray:
        """Building visual vocabulary (visual words)
        :param descriptors_lake: array of descriptors
        :param cluster_algorithm: type of cluster algorithm like K-mean, Birch
        :param k: #cluster (centroids)
        
        :return: #centroids, codebook

        """
        if cluster_algorithm.lower() in ['kmean', 'kmeans']:
            cluster_model = MiniBatchKMeans(n_clusters=k,
                                            batch_size=batch_size,
                                            random_state=123,
                                            n_init=n_init)

        # n_batches = int(len(descriptors_lake) / batch_size)
        for _ in tqdm(range(n_init), desc='Initializing'):
            cluster_model.partial_fit(descriptors_lake)

        # Final clustering
        cluster_model.fit(descriptors_lake)
        return cluster_model.cluster_centers_
    
    # ------------------------ for inference ----------------------------
    def get_embedding(
        self,
        idx: int,
        codebook: np.ndarray,
        normalized: bool = False,
        tfidf: bool = False
    ) -> np.ndarray:
        """Get embeddings of image[idx] (image > descriptors > projection in codebook > frequencies vectors)
        :param idx: int, image index
        :param codebook: np.ndarray, visual vocabulary
        :param normalized: bool, if True, normalize embedding in scale [0, 1]
        
        :return: np.array, frequencies vector (can consider as embedding)
        """
        ## What if self._get_descriptors_one_img(idx) -> None???? #TODO
        _, img_descriptors = self._get_descriptors_one_img(idx)
        if img_descriptors is not None:
            img_visual_words, distance = vq(img_descriptors, codebook)
            img_frequency_vector = np.histogram(
                img_visual_words, bins=codebook.shape[0], density=normalized
            )[0]
        else:
            img_frequency_vector = np.zeros(self.codebook.shape[1])

        if tfidf:
            # TODO
            # self._tf_idf()
            # img_frequency_vector = img_frequency_vector * self.idf
            pass
        
        return img_frequency_vector
    # ------------------------ end / for inference / section ----------------------------

    # TODO
    def _tf_idf(self):
        """TODO: Reweight important features in codebook"""
        self.idf = 1

        all_embeddings = []
        for i in range(len(self.df)):
            embedding = self.get_embedding(i)
            all_embeddings.append(embedding)

        all_embeddings = np.stack(all_embeddings)

        N = len(self.df)
        df = np.sum(all_embeddings > 0, axis=0)
        idf = np.log(N / df)

        return idf
    # TODO

    def _get_item(self, idx) -> tuple:
        """Return pair (image(arr), label)
        :param idx: index of data

        :return: tuple, (image: np.array, label)
        """
        # get path of image
        GI_dir = {"Lower GI": "lower-gi-tract", "Upper GI": "upper-gi-tract"}

        img = self.df["Video file"][idx]
        gi_tract = GI_dir[self.df["Organ"][idx]]
        classification = self.df["Classification"][idx]
        finding = self.df["Finding"][idx]
        path = f"""{self.root_dir}/{gi_tract}/{classification}/{finding}/{img}.jpg"""
        assert (
            os.path.exists(path) == True
        ), f"{path} does not exist"  # dir existance checking

        # read image
        image = np.array(Image.open(path))
        label = self.labels.index(finding)

        return image, label
    
    def _get_descriptors_one_img(self, idx, grayscale=True):
        """Extracting descriptors for each image[idx]
        :param method: method to extract features e.g. ORB, SIFT, SURF, etc
        :param idx: image index

        :return: descriptors
        :rtype: np.array
        """
        # get image
        img, _ = self._get_item(idx)
        # preprocessing: convert to grayscale for efficient computing
        if len(img.shape) == 3 and grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # descriptors extracting
        img_keypoints, img_descriptors = self.extractor.detectAndCompute(img, None)
    
        return img_keypoints, img_descriptors
