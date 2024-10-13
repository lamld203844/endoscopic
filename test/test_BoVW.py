import joblib
from BoVW import BagOfVisualWords

model = BagOfVisualWords(
    root_dir="/media/mountHDD2/lamluuduc/endoscopy/dataset/hyperKvasir/labeled-images",
    all_descriptors_dir="/media/mountHDD2/lamluuduc/endoscopy/base-code/endoscopic/checkpoints/sample_all_descriptors_sift.pkl",
    codebook_dir="/media/mountHDD2/lamluuduc/endoscopy/base-code/endoscopic/checkpoints/bovw_codebook_sift.pkl",
)


# =========== Sanity check =================================
# ====================== Unit tests =================================================
def test_attributes():
    assert model.df.shape == (10662, 4)  # dataframe
    assert len(model.labels) == 23  # #labels


# test _get_item method
def test_get_item():
    image, label = model._get_item(0)
    assert len(image.shape) == 3  # image is a 3-dimensional array (h, w, c)
    assert type(label) == int and 0 <= label <= 22  # label


# test _get_descriptors method
def test_get_descriptors():
    img_descriptors = model._get_descriptors(0)
    assert len(img_descriptors.shape) == 2


# test extract all descriptors process method
def test_extract_desciptors():
    # all_descriptors = model.extract_descriptors() # ensure output is 2d
    assert len(model.all_descriptors.shape) == 2, "Invalid extracting process"
    # assert len(model.sample_idx) == 1000, 'Invalid sampling'


# test build_codebook method
def test_build_codebook():
    assert model.codebook.shape == (model.k, 128), "Invalid building codebook process"


# test get_embedding method
def test_get_embedding():
    embedding = model.get_embedding(0)
    assert embedding.shape[0] == model.k
