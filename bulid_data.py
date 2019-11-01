import Terry_toolkit as tkit
import os
# data=[{ "keywords": "学习,学校","title": "借鉴：这篇最受欢迎校训，没有一个字讲学习",  "content": "“我知道，我不是因为偶然才来到这个世界，我是为"}]


def add_data(data,data_path='data/data.json'):
    
    tjson=tkit.Json(file_path=data_path)

    tjson.save(data)
    return   tjson.load()



def data_pre_train( tfrom=0, limit=10, data_path='data/data.json'):
    """
    from=0  #文章开始id
    limit=10 # 返回文章数目
    >>>data_pre_train(from=0, limit=10)
    
    """
 
    tjson=tkit.Json(file_path=data_path)
    data=tjson.load()
    # print(len(data))
    if len(data)>tfrom+limit:
        data=data[tfrom:tfrom+limit]
    elif len(data)<tfrom:
        print("数据过短了，存在问t")
        return []
    else:
        data=data[tfrom:]
    articles=[]
    for item in data:
        segs=[]
        try:
            segs.append(item['keywords'])
        except:
            pass
        try:
           segs.append(item['title'])
        except:
            pass
        segs=sentence_seg(item['content'])
        # print("\n".join(segs))
        article="\n".join(segs)
        articles.append(article)
    # print(len(articles))
    #z最后生成的文章列表
    # print(articles)
    return articles
def data_pre_train_file(train_path='./train.txt',   task_path='task.json'):
    """
    生成训练样本
    """
 
    tjson=tkit.Json(file_path=task_path)
    try:
        tasks=tjson.load()
        task=tasks[0]
        os.remove(task_path)
    except:
        # task=[]
        task={"tfrom":0,'limit':10}

    f1 = open(train_path,'w')
    # f1.write('hello boy!')
    articles=data_pre_train(tfrom=task['tfrom'], limit=task['limit'])
    # print(articles)
    f1.write("\n\n".join(articles))
    f1.close()
    task['tfrom']=task['tfrom']+len(articles)
    tjson.save([task])



from textrank4zh import TextRank4Keyword, TextRank4Sentence
def  sentence_seg(text):
    segs=tkit.Text().sentence_segmentation_v1(text)
    return segs

data_pre_train_file('data/train.txt',)