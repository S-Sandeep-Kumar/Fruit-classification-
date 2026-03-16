from fastai.vision.all import *
import os

def predict(image):
    model_path = r'D:\ROADMAP\Fruit Classification.pkl'
    if not os.path.exists(model_path):
        print('Error: model file not found')
        return
    
    learn = load_learner(model_path)
    if not os.path.exists(image):
        print('Error: image file not found')
        return
    
    img = PILImage.create(image)
    pred, pred_idx, probs = learn.predict(img)

    print(f'Result: {pred}')
    print(f'Confidance: {probs[pred_idx]*100:2f}')

if __name__ == '__main__':
    image = 'images.jpg'
    predict(image)