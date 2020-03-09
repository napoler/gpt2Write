from albert_pytorch import *
import sys
import time
import torch
import tkitText,tkitFile
# from bulid_data import *
from generate import *
from fun import *
import time
# print(torch.cuda.memory_allocated())
# print(torch.cuda.max_memory_allocated())
# def get_predict(text,plen,n,start,end,key=None):
#     if key ==None:

#         ttext=tkitText.Text()
#         tid=str(text)+str(plen)+str(n)
#         tid=ttext.md5(tid)
#     else:
#         tid=key

#     if start !=None:
#         start_clip=" --start "+str(start)
#     else:
#         start_clip=''
#     cmd = "python3 ./generate.py --prefix '''"+text+"''' --length " +str(plen)+" --nsamples "+str(n)+" --tid '''"+str(tid)+"''' --end "+str(end)+start_clip
#     print("开始处理: "+cmd)
#     # print(subprocess.call(cmd, shell=True))
#     if subprocess.call(cmd, shell=True)==0:
#         return get_temp(tid)['value'].get("text")
#     else:
#         return []

def get_predict(text, plen, n, start, end, key=None):
    ai = Ai()
    load_model = ai.load_model()
    args={"start":start,'end':end,'nsamples':n,'length':plen}
    try:
        data=ai.ai(text=text,args=args,key=key,load_model=load_model)

        model,_=load_model
        model.cpu()
        torch.cuda.empty_cache()
        del model
        return data
    except:
        return []
ttext=tkitText.Text()
start='[PT]'
end='[/PT]'
# text="柯基犬"
# key=ttext.md5(text)
# get_predict(text,50,3,start,end,key)

# start='[PT]'
# end='[/PT]'
data_path="/mnt/data/dev/github/scrapy/scrapy_baidu/scrapy_baidu/scrapy_baidu/data/all.json"
tt=tkitText.Text()
rankclass = classify(model_name_or_path='tkitfiles/rank',num_labels=4,device='cuda')
with open(data_path, 'r') as f:
    for i,line in enumerate(f):
        # print(it)
        if i%1000==0:
            print(i)
            # DB.pre_titles.ensureIndex({'title':"text"})
            # pass
        try:
            item = json.loads(line)
            # print(item)
            # sentences=[it['title']]
            # sentences=sentences+tt.sentence_segmentation_v1(it['content'])
            key=tt.md5(item['content'])
            pre_title=get_predict(item['content'],50,3,start,end,key)
            # print('pre_title11111111111111',pre_title)
            title_keys=[]
            for it in pre_title:
                if it.get("pt")==None:
                    continue
                elif it['pt'] not in title_keys and it['pt'] != item['title'] and len(it['pt'])>3:
                    print("预测标题",it['pt'])
                    titles=[]
                    title_keys.append(it)
                    p = rankclass.pre(it['pt'])
                    if p>0:
                        softmax=rankclass.softmax()
                        # print(p,softmax)
                        one_data={"_id":tt.md5(it['pt']),"title":it['pt'],'key':'','parent':item['_id'],'time':time.time(),'rank':p,'softmax':softmax,"state":'uncheck'}
                        # rankclass.release()
                        # del rankclass
                        # print("one_data",one_data)
                        try:
                            DB.pre_titles.insert_one(one_data)
                            pass
                        except:
                            pass

        except:
            pass
        # exit()











#     pre_title=get_predict(text,100,nsamples,start,end)
# for i in range(10):
#     with torch.no_grad():
#         rankclass = classify(model_name_or_path='tkitfiles/rank', num_labels=3, device='cuda')
#         p = rankclass.pre("柯基犬喜欢吃")
#         # softmax=rankclass.softmax()
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
# time.sleep(100)
