from fastai.vision.all import *

learn_inf = load_learner('C:\\Users\\Noah\\Desktop\\fastai\\minion recognizer\\export.pkl')


learn_inf.predict('C:\\Users\\Noah\\Desktop\\fastai\\minion recognizer\\minions\\test.png')