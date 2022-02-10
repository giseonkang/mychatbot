from utils.Preprocess import Preprocess
from utils.IntentModel import IntentModel

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags. 오류로 추가


p = Preprocess(word2index_dic='train_tools/chatbot_dict.bin',
               userdic='utils/user_dic.tsv')

intent = IntentModel(model_name='models/intent/intent_model.h5', proprocess=p)
query = "짜장면 하나랑 탕수육 주세요"
c = intent.predict_class(query)
print(intent.labels[c])