
import tkitText
import tkitFile
from random import choice
import argparse

def save_data(data,path='data/',name="train.json"):
    """
    保存数据
    """
    tkitFile.File().mkdir(path)
    data_path=path+name
    tjson=tkitFile.Json(file_path=data_path)
    tjson.save(data)

def data_pre_train_mongo_next_sentence( ):
    """
    构建下一句语料
    from=0  #文章开始id
    limit=10 # 返回文章数目
    >>>data_pre_train(from=0, limit=10)


    """

    i=0
    n=0
    data=[]
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', default=50000, type=int, required=False, help='长度限制')
    args = parser.parse_args()

    tt=tkitText.Text()
    data_json=tkitFile.Json(file_path='data.json')
    for it in data_json.auto_load():
        # print(it)
        sents=tt.sentence_segmentation_v1(it['content'])
        pre_sents=[]
        for i,sent in enumerate( sents):
            if i==0:
                one={
                    'sentence':it['title'],
                    'sentence_b':sent,
                    'label':1
                }
                data.append(one)
                rand_sent=choice(sents)
                if rand_sent !=sent:
                    one={
                    'sentence':it['title'],
                    'sentence_b':rand_sent,
                    'label':0
                }                
                data.append(one)

                pre_sents.append(it['title'])
                pre_sents.append(sent)
            else:
                pre_text="".join(pre_sents)
                one={
                    'sentence':pre_text[-200:],
                    'sentence_b':sent,
                    'label':1
                }
                data.append(one)
                rand_sent=choice(sents)
                if rand_sent !=sent:
                    one={
                    'sentence':pre_text[-200:],
                    'sentence_b':rand_sent,
                    'label':0
                }                
                data.append(one)
                pre_sents.append(sent)
            # print(len(data))
        if len(data)>args.limit:
            break
        if n%10000==0:
            # print("保存10000")
            pass
 
        n=n+1
    cut=int(len(data)*0.8)
    save_data(data[:cut],path='data/',name="train.json")
    save_data(data[cut:],path='data/',name="dev.json")
    data=[]
if __name__ == '__main__':
    data_pre_train_mongo_next_sentence()
    #--limit 长度限制
    #python bulid下一句语料.py --limit 1000