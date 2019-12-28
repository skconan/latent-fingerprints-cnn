import time
from utilities import *
from model import Autoencoder
from constansts import *

def main():
    project_dir = r"E:\Onedrive\KSIP\MachineLearning"
    dataset_dir = project_dir + "/dataset"




    result_dir = project_dir + "/cnn_ae_" + str(time.time()).replace(".","")
    model_dir = result_dir + "/model"
    pred_dir = result_dir + "/predict_result"

    for p in [result_dir, model_dir, pred_dir]:
        if not os.path.exists(p):
            os.mkdir(p)

    INPUT_TRAIN_IMAGES = get_file_path(PATH_INPUT_TRAIN_STFT)[:10000]
    INPUT_VAL_IMAGES = get_file_path(PATH_INPUT_VALID_STFT)[:1000]

    TARGET_TRAIN_IMAGES = []
    TARGET_VAL_IMAGES = []

    for fpath in INPUT_TRAIN_IMAGES:
        path = fpath.replace(PATH_INPUT_TRAIN_STFT, PATH_TARGET_SEG_STFT)
        TARGET_TRAIN_IMAGES.append(path)

    for fpath in INPUT_VAL_IMAGES:
        path = fpath.replace(PATH_INPUT_VALID_STFT, PATH_TARGET_SEG_STFT)
        TARGET_VAL_IMAGES.append(path)


    img_cols = 64
    img_rows = 64

    img_cols_result = 64
    img_rows_result = 64
    
    print_debug("Initial model")

    x_train = load_image(INPUT_TRAIN_IMAGES)
    print_debug("Load image traning input done")
    y_train = load_image(TARGET_TRAIN_IMAGES)
    print_debug("Load image target input done")
    x_val = load_image(INPUT_VAL_IMAGES)
    print_debug("Load image training validation done")
    y_val = load_image(TARGET_VAL_IMAGES)
    print_debug("Load image target validation input done")

    ae = Autoencoder(model_dir=model_dir, pred_dir=pred_dir)
    ae.train_model(x_train, y_train, x_val, y_val, epochs=1000, batch_size=50)

    
if __name__ == "__main__":
    main()