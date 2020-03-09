# from search import  *
from config import  *
import tkitText,tkitFile
import time
from tqdm import tqdm
data_path="/mnt/data/dev/github/scrapy/scrapy_baidu/scrapy_baidu/scrapy_baidu/data/all.json"
tjson=tkitFile.Json(file_path=data_path)
def save():
    its=[]
    i=0
    for it in DB.kg_content.find({}):
        # print(it)
        i=i+1
        its.append(it)
        if i%10000==0:
            tjson.save(its)
            print(i)
            its=[]
    tjson.save(its)
# save()


# # s=Search(name='Terry')
# # # s.init_search()
# # s.load()
# tt= tkitText.Text()
# with open(data_path, 'r') as f:
#     for i,line in tqdm(enumerate(f)):
#         # if i%100==0:
#         #     print(i)
#         try:
#             one = json.loads(line)
#             aid=tt.md5(one["title"]+str(one["content"]))
#             one['_id']=aid
#             one['path']=aid
#             try:
#                 s.add([one])
#             except:
#                 pass
#         except:
#             pass



rankclass = classify(model_name_or_path='tkitfiles/rank',num_labels=4,device='cuda')

# s=Search(name='Terry_sent')
# # s.init_search()
# s.load()
tt= tkitText.Text()

with open(data_path, 'r') as f:
    for i,line in tqdm(enumerate(f)):
        # print(it)
        if i%10000==0:
            print(i)
        try:
            it = json.loads(line)
            # print(it)
            sentences=[it['title']]
            sentences=sentences+tt.sentence_segmentation_v1(it['content'])
            # print(it['content'])
            datas=[]
            for sent in sentences:
                key=tt.md5(sent)
                data={'title':str(it['_id']),'content':str(sent),'path':str(key)}
                p = rankclass.pre(sent)
                if p>1:
                    softmax=rankclass.softmax()
                    # print(p,softmax)
                    one_data={"_id":tt.md5(sent),"title":sent,'key':'','parent':it['_id'],'time':time.time(),'rank':p,'softmax':softmax,"state":'uncheck'}
                    # print(one_data)
                    try:
                        DB.find_titles.insert_one(one_data)
                        pass
                    except:
                        pass
            # try:
            #     s.add(datas)
            # except:
            #     pass
        except:
            pass









print(s.find_one('小猫很可爱'))
# print(len(s.find('边境牧羊犬')))

# for i,item in enumerate(s.find('流浪狗')):
#     print(i,'####'*20)
#     print(item)

# from scrapyd_api import ScrapydAPI

# scrapyd = ScrapydAPI('http://localhost:6800')

# print(scrapyd.list_jobs('project_name'))