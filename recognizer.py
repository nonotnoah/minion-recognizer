from fastai.vision.all import *

learn_inf = load_learner('C:\\Users\\Noah\\Desktop\\fastai\\minion recognizer\\export.pkl')


pred, pred_idx, probs = learn_inf.predict('C:\\Users\\Noah\\Desktop\\fastai\\minion recognizer\\minions\\test.png')

probability = float(f'{probs[pred_idx]:.4f}') * 100 
probability = str(f'{probability:.2f}')

print(f'Prediction: {pred}; Confidence: {probability}%')