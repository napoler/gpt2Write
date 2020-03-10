
import tkitText
import tkitFile
from random import choice


def save_data(data,path='data/',name="train.json"):
    """
    保存数据
    """
    tkitFile.File().mkdir(path)
    data_path=path+name
    tjson=tkitFile.Json(file_path=data_path)
    tjson.save(data)

def read( ):
    """
    构建下一句语料
    from=0  #文章开始id
    limit=10 # 返回文章数目
    >>>data_pre_train(from=0, limit=10)


    """

    i=0
    n=0
    data=[]
    tt=tkitText.Text()
    data_json=tkitFile.Json(file_path='data/train.json')
    for it in data_json.auto_load():
        print(it)
  
read()