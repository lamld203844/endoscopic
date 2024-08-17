import joblib
from BoVW import BagOfVisualWords

model = BagOfVisualWords(
    root_dir="/media/mountHDD2/lamluuduc/endoscopy/dataset/hyperKvasir/labeled-images"
)


# =========== Sanity check =================================
# ====================== Unit tests =================================================
def test_attributes():
    assert model.df.shape == (10662, 4)  # dataframe
    assert len(model.labels) == 23


# test _get_item method
def test_get_item():
    image, label = model._get_item(0)
    assert len(image.shape) == 3  # image is a 3-dimensional array (h, w, c)
    assert type(label) == int and 0 <= label <= 23  # label


# test _feature_detecting method
def test_feature_detecting():
    img_descriptors = model._feature_detecting(0)
    assert len(img_descriptors.shape) == 2


# test extract_descriptors method
def test_extract_desciptors():
    # all_descriptors = model.extract_descriptors() # ensure output is 2d
    all_descriptors = joblib.load("all_descriptors.pkl")
    assert len(all_descriptors.shape) == 2, "Invalid extracting process"


# test build_codebook method
k, codebook = joblib.load("bovw-codebook.pkl")


def test_build_codebook():
    # codebook, variance = model.build_codebook(all_descriptors, k=k)
    assert codebook.shape == (k, 128), "Invalid building codebook process"


# test get_embedding method
def test_get_embedding():
    embedding = model.get_embedding(0, k, codebook)
    assert embedding.shape[0] == k
