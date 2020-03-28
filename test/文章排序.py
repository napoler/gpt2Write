from albert_pytorch import *
import sys
import time
import torch
import tkitText
# print(torch.cuda.memory_allocated())
# print(torch.cuda.max_memory_allocated())
from pprint import pprint
from NextSentence import *

from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl import Q
def search_content(keyword):
    client = Elasticsearch()
    q = Q("multi_match", query=keyword, fields=['title', 'body'])
    # s = s.query(q)

    # def search()
    s = Search(using=client)
    # s = Search(using=client, index="pet-index").query("match", content="金毛")
    s = Search(using=client, index="pet-index").query(q)
    response = s.execute()
    return response
    # for hit in response:
    #     print(hit.meta)
    #     print(hit.meta.score)
    #     print(hit)

def search_sent(keyword):
    client = Elasticsearch()
    q = Q("multi_match", query=keyword, fields=['title', 'content'])
    s = Search(using=client)
    # s = Search(using=client, index="pet-index").query("match", content="金毛")
    s = Search(using=client, index="pet-sent-index").query(q)
    response = s.execute()
    return response




# text="""
# 现在养狗的人越来越多，而很多人养狗之后，都会患病。今天小编就来说一下，养狗后越来越多人会患的10种病，看看你是不是全中！


# 1、容易妒忌吃醋的病

# 养狗之前，你回家，爸妈肯定是把你当宝的，而养狗之后，你回家，爸妈只会记得狗狗，完全把你忘在身后了，所以你经常都会妒忌狗狗，和狗狗吃醋。


# 2、自言自语的病

# 养狗后，你会发现自己越来越唠叨，还有了和狗狗说话的怪毛病，即便知道它不会回答你，但就是喜欢把心里话说给它听。很多一天跟狗说的话，都比和人说的话多！


# 3、一旦加班就会焦虑的病

# 养狗后，一旦加班，心情就会变得很焦虑。因为会想到狗狗自己可怜巴巴地望着窗外，一心等待自己回去，就恨不得一下班就坐火箭飞回家。加班？那是不可能的！


# 4、相思病

# 养了狗狗之后，心中挂念的永远都是家里的小家伙，就连上个班，都会惦记着狗狗吃得好不好，玩得开不开心，想不想自己等等。出个差还要托朋友照顾它，还要朋友帮忙和狗狗视频，这简直就是得了相思病。


# 5、人家晒娃我晒狗的病

# 养了狗狗之后，会发现别人晒娃，你晒狗。每天都不自觉地拿着手机拍拍拍，无论狗狗什么样都觉得可爱，统统都想记录下来，并且发到各种社交平台。如果有人评论赞了狗狗，好像心情都会变好。养狗的你，有得这种病吗？


# 6、和狗狗便便打交道的病

# 每天一睁开眼，就关心狗狗有没有排便，要是它拉了便便，就好像自己中了奖一样高兴。要是狗狗连续两天没有便便，就像天要塌下来一样，什么心情都没有，感觉比自己便秘还难受。


# 7、情绪不受控制的病

# 养了狗狗之后，你会发现自己的情绪根本不受自己的控制，全凭狗狗来控制。前一秒还在因为狗狗闯祸而生气，骂骂咧咧地，下一秒就会因为它的委屈表情、撒娇卖萌而瞬间破功，情绪根本就是随着狗狗的一举一动变化的。


# 8、宅男宅女病

# 养狗之前，宠主很喜欢和朋友聚餐、外出蹦迪的，而养狗之后，这些统统都推掉不去了。而自己这么做，没有别的原因，就是因为家里有狗，自己不忍心放下狗狗跑去疯，久而久之，就会变成宅男宅女。


# 9、抗拒旅游的病

# 养狗之后，有些人还患了抗拒旅游的病。之前时不时都会来一场说走就走的旅游，现在一旦说旅游，而且不能带狗狗的，那不用考虑了，不去！


# 10、购物狂病

# 养了狗狗之后，每次上街都是疯狂大购物，完全变成了购物狂。只要看见适合狗狗的衣服、装饰，好吃的零食和狗粮等，看见了就买，感觉钱都不是钱了。
# """
ns=NextSent()
ns.load()
text_a="这5点你都不知道,还想养柯基犬?"
# li=ns.auto_pre(text_a,text)
# # pprint(li)

# print(li)


text_a=input("输入标题：")

while True:
    print("##"*20)
    print("上文:",text_a)
    text=''
    print("##"*20)

    kw=input("输入关键词：")+text_a

    for i,it in enumerate( search_content(kw)):
    # for it in  search_content("柯基犬"):
        # print(it['title'])
        # print(it['content'])
        text=text+it['title']+"\n"
        text=text+it['content']+"\n"
        if i>10:
            break

    # li=ns.auto_pre(text_a,text)
        li=ns.auto_pre_one(text_a,text)
        

    # pprint(li[:10])
    for n,s in enumerate( li[:10]):
        print(n,s)
    print("##"*20)
    li_id=input("选择句子id：")
    sen=li[int(li_id)][0]
    text_a=text_a+sen

















# for i in range(10):
#     with torch.no_grad():
#         rankclass = classify(model_name_or_path='tkitfiles/next', num_labels=2, device='cuda')
#         p = rankclass.pre("柯基犬喜欢吃")
#         print(p)
#     rankclass.release()
    # del rankclass
#
#     # print(torch.cuda.memory_cached())
# time.sleep(1)
# model_name_or_path='tkitfiles/rank'
# P = Plus()
# P.args['class_name'] = "AlbertForSequenceClassification"
# P.args['model_name_or_path'] = model_name_or_path
# P.args['finetuning_task'] = 'finetuning_task'
# P.args['num_labels'] = 3
# print(torch.cuda.max_memory_cached())
# model,tokenizer,config_class=P.load_model()
# print(torch.cuda.memory_cached())
# P.release()
# torch.cuda.empty_cache()
# print(torch.cuda.memory_cached())
# time.sleep(100)

# device = torch.device('cuda:0')
# # with torch.no_grad():
# # 定义两个tensor
# dummy_tensor_4 = torch.randn(120, 3, 512, 512).float().to(device)  # 120*3*512*512*4/1000/1000 = 377.48M
# dummy_tensor_2 = torch.randn(80, 3, 512, 512).float().to(device)  # 80*3*512*512*4/1000/1000 = 251.64M
# print(torch.cuda.memory_cached())
# time.sleep(1)
# # 然后释放
# dummy_tensor_4 = dummy_tensor_4.cpu()
# dummy_tensor_2 = dummy_tensor_2.cpu()
# # 这里虽然将上面的显存释放了，但是我们通过Nvidia-smi命令看到显存依然在占用
# print(torch.cuda.memory_cached())
# torch.cuda.empty_cache()
# # 只有执行完上面这句，显存才会在Nvidia-smi中释放
# print(torch.cuda.memory_cached())
time.sleep(100)