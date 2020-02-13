

from config import *
import  requests
import  json
import time
from random import choice
def get():
    response = requests.get(
        'http://localhost:6800/listjobs.json?project=default',
        # params={'keyword': keyword,'project':'default','spider':'kgbot',"url_type":'bytime'},
        params={'project':'default'}
    )
    if response.status_code ==200:
        # print(it)
        return response.json()
        pass
# print(get()['running'])

# print(get()['pending'])

def run(url_type='bytime'):
    keywords=[]
    items=DB.pre_titles.aggregate([{ '$group': { '_id' : '$key', 'count': { '$sum' : 1 } } }, { '$match': { 'count': { '$gt' : 1} } }])
    for it in items:
        keywords.append(it['_id'])
    print("keywords1",len(keywords)) 
    items=DB.keywords.find({})
    for it in items:
        # print(it)
        keywords.append(it['_id'])

    keywords=list(set(keywords))
    print("keywords1",len(keywords)) 
    for it in keywords:
        print("关键：",it)
        n=len(get()['running'])+len(get()['pending'])
        while n>10:
            time.sleep(30)
            
            n=len(get()['running'])+len(get()['pending'])
            

        response = requests.post(
            'http://localhost:6800/schedule.json',
            # params={'keyword': keyword,'project':'default','spider':'kgbot',"url_type":'bytime'},
            params={'keyword': it,'project':'default','spider':'kgbot',"url_type":url_type},
        )
        if response.status_code ==200:
            # print(it)
            pass
            # print("提交进程",response.json())

while True:
    url_type=choice(['bytime','all',''])
    print("url_type",url_type)
    run(url_type=url_type)
    time.sleep(60*60) #休息时间


# # 批量添加关键词
# f='/mnt/data/dev/github/Terry-toolkit/Terry-toolkit/Terry_toolkit/resources/THUOCL.json'
# with open(f, 'r') as f:
#     data = json.load(f)
#     # print(data['动物'])
#     for it in data['动物']:
#         try:
#             DB.keywords.insert_one({"_id":it,'value':it})
#             print('添加关键词',it)
#         except :
#             pass