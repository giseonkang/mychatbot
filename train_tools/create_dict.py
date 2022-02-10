import sys
sys.path.insert(0, 'C:/Users/82104/Desktop/chatbot')

from utils.Preprocess import Preprocess
from tensorflow.keras import preprocessing
import pickle

# 말뭉치 데이터 읽어오기
def read_corpus_data(filename):
    with open(filename, 'r',encoding='UTF-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data


# 말뭉치 데이터 가져오기
corpus_data = read_corpus_data('./train_tools/corpus.txt')


# 망뭉치 데이터에서 키워드만 추출해서 사전 리스트 생성
p = Preprocess(word2index_dic='./train_tools/chatbot_dict.bin',
               userdic = 'utils/user_dic.tsv')
dict = []
for c in corpus_data:
    pos = p.pos(c[1])
    for k in pos:
        dict.append(k[0])



tokenizer = preprocessing.text.Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(dict)
word_index = tokenizer.word_index

# 사전 파일 생성
f = open("chatbot_dict.bin", "wb")
try:
    pickle.dump(word_index, f)
except Exception as e:
    print(e)
finally:
    f.close()