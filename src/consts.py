INPUT_IMAGES_FOLDER = '../images/Images/'
PREPROCESSED_IMAGES_FOLDER = '../images_scaled5/'

DATA_FOLDER = '../data/'

DOG_BREEDS_FN = 'dog_breeds5.csv'

# TEST_IMAGE_WIDTH = 443
# TEST_IMAGE_HEIGHT = 386

TEST_IMAGE_WIDTH = 256
TEST_IMAGE_HEIGHT = 256

SEED_VALUE = 42

BATCH_SIZE = 32
EPOCH_NUMS = 200

MODEL_PATH = '../data/full_model'
WEIGHTS_PATH = '../data/model_weights'

LABEL_INDEXES = '../data/labels.csv'


GEN_TRAIN_IMAGES = '../data/gen_train_images'
GEN_VALIDATE_IMAGES = '../data/gen_validate_images'


TRAIN_TRANSFORM_DIR = '../images/images5_train'
TEST_TRANSFORM_DIR = '../images/images5_validate'

BOTTLENECK_FEATURES_TRAIN = '../data/bottleneck_features_train.npy'
BOTTLENECK_FEATURES_TEST = '../data/bottleneck_features_validation.npy'

BOTTLENECK_WEIGTHS = '../data/bottleneck_fc_weights.h5'
BOTTLENECK_FULL_MODEL = '../data/bottleneck_full_model.h5'


FINETUNE_WEIGHTS = '../data/finetuned_wegihts.h5'
FINETUNE_FULL_MODEL = '../data/finetuned_full_model.h5'