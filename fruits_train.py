import kagglehub
from fastai.vision.all import *
import os

def train_model():
    save_dir = r'D:\ROADMAP'
    model_name = 'Fruit Classification.pkl'
    file = os.path.join(save_dir, model_name)
    path = kagglehub.dataset_download("shivamardeshna/fruits-dataset")

    dls = ImageDataLoaders.from_folder(path, valid_pct = 0.2, item_tfms = Resize(224))
    learn = vision_learner(dls, resnet34, metrics = error_rate)

    learn.fine_tune(12)
    learn.export(file)

if __name__ == "__main__":
    train_model()