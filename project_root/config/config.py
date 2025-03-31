ROOT_DIR_PATH = "~/mini_projects/tumor_classification/project_root"
FILENAME = "breast-cancer-wisconsin-data.csv"
DATA_DIR = "dataset"



TARGET_COLUMN  = "diagnosis"


REFINED_COLUMNS = ['diagnosis',
 'radius_mean',
 'perimeter_mean',
 'area_mean',
 'compactness_mean',
 'concavity_mean',
 'concave points_mean',
 'radius_se',
 'perimeter_se',
 'area_se',
 'radius_worst',
 'perimeter_worst',
 'area_worst',
 'compactness_worst',
 'concavity_worst',
 'concave points_worst']


DEGREE = 1


IS_ONLY_INTERACTION = False

TRAINING_DATA_FRAC = 0.7
TESTING_DATA_FRAC = 0.3

EPSILON = 10**(-3)
TOL = 10**(-7)


TRAINING_DATA_FILENAME = "training_data.csv"
TESTING_DATA_FILENAME = "testing_data.csv"

SAVED_MODEL_FILE = "trained_model.pkl"
SAVED_MODEL_PATH = "models"