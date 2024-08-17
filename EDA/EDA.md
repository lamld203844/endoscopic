# HyperKvasir: Exploratory Data Analysis
**General:** 

gastrointestinal tract image dataset : 110,079 images and 373 videos where it captures anatomical landmarks and pathological and normal findings. The results is more than 1,1 million images and video frames all together.

![general_HyperKvasir.drawio.png](../asset/dataset-hierachy.png)

## Labeled images

- 10,662 labeled images JPEG format.
- 23 different classes. number of images per class are imbalanced
- **Imbalance classes**
    
    ![Untitled](../asset/labeled_img-distribution.png)
    
- **Image sizes**
    - Varying - 751 different sizes / 10662 images
        
        ![Untitled](../asset/labeled_img-size_distribution.png)
        
    - Number of top n sizes vs %total images
        
        ![Untitled](../asset/size_vs_img.png)
        
- Visualization each class and corresponding images
    
    ![Untitled](../asset/labeled_img.png)
    

## Unlabeled images

99,417 unlabeled images

![Untitled](../asset/unlabeled_img.png)

## Segmented images

1,000 images from the polyp class (original - mask - bounding box)

- segmentation mask: ROI = the pixels depicting polyp tissue = (white mask), others in black.
- bounding box: outermost pixels of the found polyp.

![Untitled](../asset/segmented_img.png)

## Annotated video

 373 videos (different findings and landmarks) approximately 11.62 hours of videos and 1,059,519 video frames

that can be converted to images if needed. Each video has been manually assessed by a medical professional working in the field of gastroenterology and resulted in a total of 171 annotated findings.