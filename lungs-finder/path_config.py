DICOM_TRAIN_DATASET = r"C:\Users\m\Desktop\datasets\dicom_train"
PNG_TRAIN_DATASET = r"C:\Users\m\Desktop\datasets\lungs_train_2000" #COLAB_PATH "/content/lungs_train_2000/lungs_train_2000"

TENSORBOARD_PATH = r"C:\Users\m\Desktop\LUNGS\lungs-finder\logs"

BATCH_SIZE = 10

### OBJECT DETECTION
N_PROPOSALS = 5

class TaskType:
    BINARY_CLASSIFICATION = "BINARY_CLASSIFICATION"
    MULTICLASS_CLASSIFICATION = "MULTICLASS_CLASSIFICATION"
    OBJECT_DETECTION = "OBJECT_DETECTION"