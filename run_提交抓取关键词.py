

from config import *
import  requests
import  json
keywords=[]
# items=DB.pre_titles.find({'key':keyword})

items=DB.pre_titles.aggregate([{ '$group': { '_id' : '$key', 'count': { '$sum' : 1 } } }, { '$match': { 'count': { '$gt' : 1} } }])
for it in items:
    keywords.append(it['_id'])
print("keywords1",len(keywords)) 
items=DB.keywords.find({})
for it in items:
    print(it)
    keywords.append(it['_id'])

keywords=list(set(keywords))
print("keywords1",len(keywords)) 
for it in keywords:
    response = requests.post(
        'http://localhost:6800/schedule.json',
        # params={'keyword': keyword,'project':'default','spider':'kgbot',"url_type":'bytime'},
        params={'keyword': it,'project':'default','spider':'kgbot',"url_type":'bytime'},
    )
    if response.status_code ==200:
        # print(it)
        pass
        # print("提交进程",response.json())


# 批量添加关键词
# f='/mnt/data/dev/github/Terry-toolkit/Terry-toolkit/Terry_toolkit/resources/THUOCL.json'
# with open(f, 'r') as f:
#     data = json.load(f)
#     # print(data['动物'])
#     for it in data['动物']:
#         try:
#             DB.keywords.insert_one({"_id":keyword,'value':keyword})
#             print('添加关键词',keyword)
#         except :
#             pass