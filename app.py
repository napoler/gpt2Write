
from flask import Flask, render_template, request, json, Response, jsonify,escape
# from generate import *
from bulid_data import *
import tkitDb,tkitText,tkitFile,tkitNlp
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import gc
import subprocess
import  requests
from pprint import  pprint
from flask_socketio import SocketIO, emit
import jieba.analyse
import tkitW2vec
import tkitMarker,tkitDb,tkitText
import os
from fun import *



from config import *
import gc
gc.set_threshold(700, 10, 5)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)




"""
默认的限制器
key_func参数是判断函数,表示以何种条件判断算一次访问?这里使用的是get_remote_address,此函数返回的是客户端的访问地址.
default_limits 是一个数组,用于依次提同判断条件.比如100/day是指一天100次访问限制.
常用的访问限制字符串格式如下:
10 per hour
10/hour
10/hour;100/day;2000 per year
100/day, 500/7days
注意默认的限制器对所有视图都有效,除非你自定义一个限制器用来覆盖默认限制器,或者使用limiter.exempt装饰器来取消限制
"""
limiter = Limiter(app=app, key_func=get_remote_address, default_limits=["1000000/day, 60/minute, 1/second"])



def get_post_data():
    """
    从请求中获取参数
    :return:
    """
    data = {}
    try:
        if request.content_type.startswith('application/json'):
            data = request.get_data()
            data = json.loads(data)
        else:
            for key, value in request.form.items():
                if key.endswith('[]'):
                    data[key[:-2]] = request.form.getlist(key)
                else:
                    data[key] = value
    except :
        pass
    return data
def get_predict(text,plen,n,start,end,key=None):
    if key ==None:

        ttext=tkitText.Text()
        tid=str(text)+str(plen)+str(n)
        tid=ttext.md5(tid)
    else:
        tid=key

    if start !=None:
        start_clip=" --start "+str(start)
    else:
        start_clip=''
    cmd = "python3 ./generate.py --prefix '''"+text+"''' --length " +str(plen)+" --nsamples "+str(n)+" --tid '''"+str(tid)+"''' --end "+str(end)+start_clip
    print("开始处理: "+cmd)
    # print(subprocess.call(cmd, shell=True))
    if subprocess.call(cmd, shell=True)==0:
        return get_temp(tid)['value'].get("text")
    else:
        return []
# def get_predict(text, plen, n, start, end, key=None):
#     ai = Ai()
#     load_model = ai.load_model()
#     args={"start":start,'end':end,'nsamples':n,'length':plen}
#     try:
#         data=ai.ai(text=text,args=args,key=key,load_model=load_model)

#         model,_=load_model
#         model.cpu()
#         torch.cuda.empty_cache()
#         del model
#         return data
#     except:
#         return []


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/edit')
def edit():
    return render_template("edit.html")

@app.route('/titles',methods=['GET', 'POST'])
def titles():
    page = request.args.get('page')
    state = request.args.get('state')
    q={}
    skip=0
    limit=20
    if page==None or page=='':
        page=0
        skip=0
    else:
        skip=limit*int(page)
        page=int(page)
        pass
    if state==None or state=='':
        # items=DB.pre_titles.find({})
        pass
    else:
        # items=DB.pre_titles.find({"state":state})
        q['state']=state
        pass
    items=DB.pre_titles.find(q).limit(limit).skip(skip).sort('time',-1)
    # print(items)
    data={
        'page':page,
        'nextpage':page+1,
        'prepage':page-1,
        'state':state,
        'items':items

    }
    return render_template("titles.html",data=data)    
# 获取预测结果
@app.route("/json/predict",methods=['GET', 'POST'])
# 自定义限制器覆盖了默认限制器
@limiter.limit("100/minute, 1/second")
def json_predict():
    # #句子

    data= get_post_data()
    print('data',data)
    # paragraph = request.args.get('text')
    # previous_line=request.args.get('sentence')
    text = data['text']
    plen= data['plen']
    n=data['n']
    # text_array={'original':text,
    # 'items':get_predict(text,plen,n)
    # }
    # args={"end":"[/PT]",'start':'[PT]'}
    # print(get_writing("柯基犬",args))
    # p=get_writing("柯基犬",args)
    start=None
    end='[END]'
    get_pre=get_predict(text,plen,n,start,end)
    print("get_pre",get_pre)
    try:
        text_list=[]

        for it in get_pre:
            text_list.append(it['all'])

        text_array={'original':text,
        'items':text_list
        }
    except :
        text_array={'original':text,
        'items':[]
        }
    
    
    # for x in locals().keys():
    #     # print("清理函数内存")
    #     del locals()[x]
    # gc.collect()
    return jsonify(text_array)


# def get_predict(text,plen,n):
#     ttext=tkitText.Text()
#     tid=str(text)+str(plen)+str(n)
#     tid=ttext.md5(tid)

#     data_path="tmp/run_task"+tid+".json"
#     print('load',data_path)
#     if not os.path.exists(data_path):
#         # 不存在缓存，重新预测
#         cmd = "python3 ./generate.py --prefix '''"+text+"''' --length " +str(plen)+" --nsamples "+str(n)+" --tid "+str(tid)
#         print("开始处理: "+cmd)
#         # print(subprocess.call(cmd, shell=True))
#         if subprocess.call(cmd, shell=True)==0:

#             try:
#                 tjson=tkitFile.Json(file_path=data_path)
#                 return   tjson.load()[0]['data']
#             except:
#                 print('load文件失败',data_path)
#                 return{}
#                 pass
#         else:
#             return{}
#     else:
#         #加载缓存预测
#         try:
#             tjson=tkitFile.Json(file_path=data_path)
#             return   tjson.load()[0]['data']
#         except:
#             print('load文件失败',data_path)
#             return{}


@app.route("/explore",methods=['GET'])
def explore():
    """
    构建训练数据
    """

    return render_template("explore.html") 

@app.route("/search",methods=['GET'])
def search():
    """
    构建训练数据
    """

    return render_template("search.html") 




            
# @app.route("/json/search",methods=['GET'])
# def json_search():
#     """
#     构建训练数据
#     """
#     keyword = request.args.get('keyword')
#     # response = requests.get(
#     #     'http://0.0.0.0:6801/json/keyword',
#     #     params={'keyword': keyword,'limit':4},
#     # )
#     # http://localhost:6800/schedule.json&project=default&spider=kgbot&setting=DOWNLOAD_DELAY=2&keyword="宠物狗"
#     response = requests.post(
#         'http://localhost:6800/schedule.json',
#         params={'keyword': keyword,'project':'default','spider':'kgbot'},
#     )
#     if response.status_code ==200:
#         print("提交进程",response.json() )

#     response = requests.get(
#         'http://0.0.0.0:6801/json/search',
#         params={'keyword': keyword,'limit':4},
#     )
    
#     # tt=tkitText.Text()
#     # pprint({'keyword': keyword,'limit':20})
#     if response.status_code ==200:
#         items=response.json() 
#         titles=[]
#         print("获取数据数目:",len(items))
#         for item in items:
#             # text=" [tt] "+item['title']+" [/tt] "+item['content']
#             text=item['content']
#             text=text[:300]+" [pt]"
#             ai_title(text=text,key=keyword)
#             # titles=titles+ai_title(text=text,key=keyword)
#             # print('req',req)
#         data={'items':items,"titles":titles,"limit":20}
#     return jsonify(data)

            
@app.route("/json/get/title",methods=['GET'])
def json_get_title():
    """
    构建训练数据
    """
    keyword = request.args.get('keyword')
    items=DB.pre_titles.find({'key':keyword})
    titles=[]
    for item in items:
        print(item)
        titles.append({'title':item['title']})
    data={"items":titles,"limit":20}
    return jsonify(data)

def get_predict_title(text,plen,n):
    ttext=tkitText.Text()
    tid=str(text)+str(plen)+str(n)
    tid=ttext.md5(tid)

    data_path="tmp/run_task"+tid+".json"
    print('load',data_path)
    if not os.path.exists(data_path):
        # 不存在缓存，重新预测
        cmd = "python3 ./generate.py --prefix '''"+text+"''' --length " +str(plen)+" --nsamples "+str(n)+" --tid "+str(tid)
        print("开始处理: "+cmd)
        # print(subprocess.call(cmd, shell=True))
        if subprocess.call(cmd, shell=True)==0:

            try:
                tjson=tkitFile.Json(file_path=data_path)
                return   tjson.load()[0]['data']
            except:
                print('load文件失败',data_path)
                return{}
                pass
        else:
            return{}
    else:
        #加载缓存预测
        try:
            tjson=tkitFile.Json(file_path=data_path)
            return   tjson.load()[0]['data']
        except:
            print('load文件失败',data_path)
            return{}
            


# @app.route("/json/pre/title",methods=['POST'])
# @limiter.limit("100/minute, 1/second")    
# def json_pre_title():
#     """
#     构建训练数据
#     """
#     data = get_post_data()
#     print(data)
#     text=" [/tt] "+data.get('title')+" [/tt] "+data.get('content')
#     text=text[:300]+" [pt]"
#     req=ai_title(text=text)
#     print(req)
    
#     rdata={"items":[],"num":2}
#     return jsonify(rdata)












        
    

   

@app.route('/add/data')
def add_data_page():
    return render_template("add_data.html")
#
@app.route("/json/add/data",methods=['GET', 'POST'])
def json_add_data():
    """
    构建训练数据
    """
    data= get_post_data()
    
    new_data=[]
    new_data.append(data)
    print('data',new_data)
    last_data=add_data(new_data)
    text_array={"length":len(last_data),'data_last':last_data[-10:]}

    return jsonify(text_array)


@app.route("/json/pre/add/keyword",methods=[ 'POST'])
def json_pre_add_keyword():
    """
    自动预测知识
    """
    data= get_post_data()
    keywords = data.get('keywords')
    print(keywords)
    

    return 'jsonify(text_array)'

#
@app.route("/json/pre/kg",methods=['GET'])
def json_pre_kg():
    """
    自动预测知识
    """
    keyword = request.args.get('keyword')
    # http://0.0.0.0:6801/json/keyword?keyword=%E9%AC%A3%E7%8B%97&limit=1000
    # r = requests.get('https://api.github.com/user', auth=('user', 'pass'))
    # Search GitHub's repositories for requests
    # keyword="花千骨"
    response = requests.get(
        'http://0.0.0.0:6801/json/keyword',
        params={'keyword': keyword,'limit':20},
    )
    tt=tkitText.Text()
    if response.status_code ==200:
        items=response.json() 
        for item in items:
            for s in tt.sentence_segmentation_v1(item['content']):
                print(s)
                get_kg(s)
        # pprint(items)


        return jsonify(response.json())
    else:
        return ''




    return 'jsonify(text_array)'






@app.route("/json/get/keywords",methods=[ 'POST'])
def json_get_keywords():
    """
    构建训练数据
    """
    data= get_post_data()
    ttext=tkitText.Text()
    keywords=ttext.get_keywords(data['text'],num=40)
    return jsonify(keywords)

@app.route("/json/get/marker",methods=[ 'POST'])
def json_get_ner():
    """
    对文本进行标记 发现实体 用来获取知识信息
    """
    data= get_post_data()
    P=get_ner()
    
    # print(result)
    # keywords=ttext.get_keywords(data['text'],num=40)
    db=tkitDb.LDB(path="/mnt/data/dev/github/标注数据/Bert-BiLSTM-CRF-pytorch/tdata/lvkg.db")
    db.load("kg")
    # print("result",result)
 
    kgs=[]
    all_words=[]
    tt=tkitText.Text()
    sentences=tt.sentence_segmentation_v1(data['text'])

    nlp=Nlp()
    words_list=[]
    for sentence in sentences:
        words_list=words_list+nlp.ner(sentence)
    all_words=[]
    new_words_list=[]
    for word in words_list:
        #去除重复关键词
        if word in all_words:
            continue
            pass
        else:
            all_words.append(word)
        try:
            kg=db.get(word)
            # print(kw['words'],db.str_dict(kg))
            # kg=db.get('犬')
            kgs.append({'word':word,'type':'实体','kg':db.str_dict(kg)})
            print("知识获取成功")
        except:
            kgs.append({'word':word,'type':'实体','kg':{}})
            pass
    # print(words_list)
    # print("kk",kgs)
    print("ltp发现实体数目",len(words_list))
    result=P.pre(sentences)
    for t,kws in result:
        for kw in kws:
            # print(kw)

            #去除重复关键词
            if kw['words'] in all_words:
                # continue
                pass
            else:
                all_words.append(kw['words'])
            
            if kw['type']=="实体":
                try:
                    kg=db.get(kw['words'])
                    # print(kw['words'],db.str_dict(kg))
                    
                    # kg=db.get('犬')
                    kgs.append({'word':kw['words'],'type':kw['type'],'kg':db.str_dict(kg)})
                    print("知识获取成功")
                except:
                    kgs.append({'word':kw['words'],'type':kw['type'],'kg':{}})
                    pass
            elif kw['type']=="描述":
                kgs.append({'word':kw['words'],'type':kw['type'],'kg':{}})
            elif kw['type']=="关系":
                kgs.append({'word':kw['words'],'type':kw['type'],'kg':{}})
                pass
            else:
                kgs.append({'word':kw['words'],'type':kw['type'],'kg':{}})
                pass
    print("总共发现实体数目",len(kgs))
    print(kgs)
    return jsonify(kgs)





@app.route("/json/get/keyseq",methods=[ 'POST'])
def json_get_keyseq():
    """
    构建训练数据
    """
    data= get_post_data()
    # ttext=tkit.Text()
    seq=get_keyseq(data['text'],num=30)
    return jsonify(seq)



@app.route("/json/bulid/train",methods=[ 'POST'])
def json_bulid_train():
    """
    构建训练数据
    """
    # data= get_post_data()
    # ttext=tkit.Text()
    # keywords=ttext.get_keywords(data['text'],num=10)
    data=data_pre_train_file()
    return jsonify(data)
import time

# from gensim.models import KeyedVectors
# from threading import Semaphore

# # m=gensim.models.word2vec.Word2Vec.load(Word2vec_model_save_fast,mmap='r')
# model = KeyedVectors.load(Word2vec_model_save_fast, mmap='r')
# model.syn0norm = model.syn0  # prevent recalc of normed vectors
# print(model.most_similar("柯基犬"))
# Semaphore(0).acquire()  # just hang until process killed



# @app.route("/json/search",methods=['GET'])
@socketio.on('关键词探索', namespace='/tapi')
def wordexplore(message):
    """
    构建训练数据
    """
    print('message',message)
    keyword = message.get('data')
    url_type = message['url_type']
    print('url_type',url_type)
    w2vWV=tkitW2vec.Word2vec()
    # # # w2v.load(model_file=Word2vec_model)
    # # w2v.load(model_file=Word2vec_model_WV,model_tpye='wv',binary=True)
    # w2vWV=tkitW2vec.Word2vec()
    # # w2v.load(model_file=Word2vec_model)
    w2vWV.load(model_file=Word2vec_model_save_fast,model_tpye='fast')

    # w='小野猫'
    # print(w,"的同义词")
    kws=w2vWV.most_similar(keyword)
    # print(len(kws))
    keywords=[]
    for word,rank in kws:
        # print(word,"----》",rank )
        keywords.append(word)
    emit('预测反馈', {'state': 'success','step':'wordexplore','data':keywords})
    # del w2v
    del kws
    del w2vWV
    
    gc.collect()

    response = requests.post(
        'http://localhost:6800/schedule.json',
        params={'keyword': keyword,'project':'default','spider':'kgbot',"url_type":url_type},
        )
    try:
        DB.keywords.insert_one({"_id":keyword,'value':keyword})
        print('添加关键词',keyword)
    except :
        pass
    if response.status_code ==200:
        print("提交进程",response.json())
        #修改状态为提交搜索成功
        emit('预测反馈', {'state': 'success','step':'add','data':response.json()})
    else:
        emit('预测反馈', {'state': 'fail','step':'add','data':response.json()})
    # Semaphore(0).acquire()  # just hang until process killed



@socketio.on('添加标题', namespace='/tapi')
def addtitle(message):
    """
    构建训练数据
    """
    data = message.get('data')
    # tt=tkitText.Text()
    
    try:

        DB.pre_titles.update_one({'_id':data["_id"]},   {"$set" :{'state':'good'}}) 
    except :
        pass
    emit('预测反馈', {'state': 'success','step':'addtitle','data':data})
    

@socketio.on('获取摘要', namespace='/tapi')
def get_sumy_text(message):
    keyword = message.get('data')
 
    print(("keyword",keyword))
    response = requests.post(
        'http://localhost:6800/schedule.json',
        params={'keyword': keyword,'project':'default','spider':'kgbot',"url_type":'all'},
        )
    try:
        DB.keywords.insert_one({"_id":keyword,'value':keyword})
        print('添加关键词',keyword)
    except :
        pass
    if response.status_code ==200:
        print("提交进程",response.json())

    response = requests.get(
        'http://0.0.0.0:6801/json/search_sent',
        params={'keyword': keyword,'limit':50},
    )
    if response.status_code ==200:
        items=response.json()
        for  item in items:
            print(item)

            # print(l)
            data=item
            # data['sumy']=l
            # data['content_list']=s
            emit('搜索句子', {'state': 'success','step':'search_sent','data':data})
    # 请求进程结果
    response = requests.get(
        'http://0.0.0.0:6801/json/search',
        params={'keyword': keyword,'limit':50},
    )
    if response.status_code ==200:
        items=response.json()
        for  item in items:
            print(item)
            l,s=get_sumy(item['content'])
            # print(l)
            data=item
            data['sumy']=l
            data['content_list']=s
            emit('预测摘要', {'state': 'success','step':'sumy','data':data})
        pass


# @app.route("/json/search",methods=['GET'])
@socketio.on('添加关键词', namespace='/tapi')
def addword(message):
    """
    构建训练数据
    """
    print('message',message)
    keyword = message.get('data')
    url_type = message['url_type']
    print('url_type',url_type)
    w2vWV=tkitW2vec.Word2vec()
    # w2v.load(model_file=Word2vec_model)
    w2vWV.load(model_file=Word2vec_model_WV,model_tpye='wv',binary=True)

    # w='小野猫'
    # print(w,"的同义词")
    kws=w2vWV.most_similar(keyword)
    # print(len(kws))
    keywords=[]
    for word,rank in kws[:20]:
        # print(word,"----》",rank )
        keywords.append(word)
    emit('预测反馈', {'state': 'success','step':'keywords','data':keywords})
    # del w2v
    del kws
    gc.collect()

    response = requests.post(
        'http://localhost:6800/schedule.json',
        params={'keyword': keyword,'project':'default','spider':'kgbot',"url_type":url_type},
        )
    try:
        DB.keywords.insert_one({"_id":keyword,'value':keyword})
        print('添加关键词',keyword)
    except :
        pass
    if response.status_code ==200:
        print("提交进程",response.json())
        #修改状态为提交搜索成功
        emit('预测反馈', {'state': 'success','step':'add','data':response.json()})
    else:
        emit('预测反馈', {'state': 'fail','step':'add','data':response.json()})


# def log(value):
#     """
#     简单日志
#     """
#     vlog="处理搜索没有结束等待加载中10s"
#     emit('预测反馈', {'state': 'success','step':'log','data':{'value':vlog}}) 
# @app.route("/json/search",methods=['GET'])
@socketio.on('预测标题', namespace='/tapi')
def json_search(message):
    """
    构建训练数据
    """
    print('message',message)
    keyword = message.get('data')

    print("关键词",keyword)
    # log("添加关键词")
    emit('预测反馈', {'state': 'success','step':'log','data':"添加关键词 "+keyword}) 
    tt=tkitText.Text()
    set_var(keyword,{'do':'start'})

    items=DB.pre_titles.find({'key':keyword})
    titles=[]
    title_keys=[]
    rankclass = classify(model_name_or_path='tkitfiles/rank', num_labels=3)
    for item in items:
        print(item)

        p = rankclass.pre(item['title'])
        softmax=rankclass.softmax()

        # del rankclass
        item['softmax']=softmax
        item['rank']=p
        # titles.append({"key":item['_id'],"title":item['title'],'rank':p,'softmax':softmax})
        titles.append(item)
        if item['title'] not in title_keys:
            title_keys.append(item['title'])
    # data={"items":titles,"limit":20}
    emit('预测反馈', {'state': 'success','step':'get_titles','data':titles})
    rankclass.release()

    # response = requests.get(
    #     'http://0.0.0.0:6801/json/keyword',
    #     params={'keyword': keyword,'limit':4},
    # )
    # http://localhost:6800/schedule.json&project=default&spider=kgbot&setting=DOWNLOAD_DELAY=2&keyword="宠物狗"
    response = requests.post(
        'http://localhost:6800/schedule.json',
        # params={'keyword': keyword,'project':'default','spider':'kgbot',"url_type":'bytime'},
        params={'keyword': keyword,'project':'default','spider':'kgbot'},
    )
    try:
        DB.keywords.insert_one({"_id":keyword,'value':keyword})
    except :
        pass
    if response.status_code ==200:
        print("提交进程",response.json())
        jobid=response.json()['jobid']
        #修改状态为提交搜索成功
        emit('预测反馈', {'state': 'success','step':'add','data':response.json()})
    else:
        emit('预测反馈', {'state': 'fail','step':'add','data':response.json()})
        jobid=None
        
    keys=[]
    i=0
    jobid_finished=[]
    #循环两次第一次用于预测之前存在的文本
    # for i in range(200):
    while jobid not  in jobid_finished:
        items=[]
        emit('预测反馈', {'state': 'success','step':'log','data':"获取搜索次数 "+str(i)}) 
        #对于第一次循环获取文章过少的进行停止让后台爬虫结束，当然也可以去查询
        #这样自定义修整时间
        jobid_finished=[]
        # while  jobid not  in jobid_finished and i>0 and jobid!=None:
        if jobid!=None and i>0:
            print("处理搜索没有结束等待加载中")

            time.sleep(1)
            response = requests.get(
                'http://0.0.0.0:6800/listjobs.json',
                params={'project': 'default'},
            ) 
            print("获取数据",response.status_code,{'keyword': keyword,'limit':4},)
            if response.status_code ==200:
                # print
                # emit('预测反馈', {'state': 'success','step':'log','data':"爬虫进度 "+str(response.json())}) 
               
                for it in response.json().get('finished'):
                    jobid_finished.append(it['id'])
                    #请求搜索结果
            if len(jobid_finished)==0:
                emit('预测反馈', {'state': 'success','step':'log','data':"爬虫工作中"}) 
            else:
                emit('预测反馈', {'state': 'success','step':'log','data':"爬虫结束工作"}) 
            # 请求进程结果
            response = requests.get(
                'http://0.0.0.0:6801/json/keyword',
                params={'keyword': keyword,'limit':50},
            )
            if response.status_code ==200:
                items=response.json()
            
        emit('预测反馈', {'state': 'success','step':'log','data':"爬虫工作结束！ "}) 
        # if i==0:

        #请求搜索结果
        if len(items)==0:
            response = requests.get(
                'http://0.0.0.0:6801/json/search',
                params={'keyword': keyword,'limit':10},
            )
            print("获取数据",response.status_code,{'keyword': keyword,'limit':5},)
            if response.status_code ==200:
                # print
                items=response.json()
        
        # print("获取数据items",items)

        # titles=[]
        print("获取数据数目:",len(items))
        emit('预测反馈', {'state': 'success','step':'log','data':"获取数据数目: "+str(len(items))}) 
        if len(items)>10:
            nsamples=1
        elif 5<len(items)<10:
            nsamples=2
        else:
            nsamples=3
        for item in items:
            # print("item",item)
            text=item['content']
            keywords=jieba.analyse.extract_tags(text, topK=10, withWeight=False, allowPOS=())
            # keywords=[]
            # w2v=tkitW2vec.Word2vec()
            # w2v.load(model_file=Word2vec_model)
            # kws=w2v.keywords(text)
            # # print('kws[:10]',kws[:10])
            # # print("排名前十的关键词")
            # for word,rank in kws[:10]:
            #     keywords.append(word)
            keywords=list(set(keywords))
            emit('预测反馈', {'state': 'success','step':'keywords','data':keywords})
            del keywords
            # del kws
            # del w2v
            gc.collect()
            # print(item)
            # key=tt.md5(text)
            key=item['path']
            if key not in keys:
                item['key']=key
                emit('预测反馈', {'state': 'success','step':'get_articles','data':[item]})
                keys.append(key)
                emit('预测反馈', {'state': 'success','step':'log','data':"Ai预测标题中("+str(len(keys))+")"+str(i)}) 
                pre_data= get_var(keyword)
                # if pre_data['value'].get('do')=="stop":
                #     print("已经停止")
                #     break
                text=text[:400]
                # pre_title=ai_title(text=text,key=keyword,nsamples=nsamples)
                start='[PT]'
                end='[/PT]'
                pre_title=get_predict(text,100,nsamples,start,end)
                # print("pre_title",pre_title)
                # print(type(pre_title['text']))
                # if pre_title['text']==None:
                #     continue
                # print("pre_title",pre_title)
                # if "text" in pre_title.keys():
                #     pass
                # else:
                #     continue
                for it in pre_title:
                    if it.get("pt")==None:
                        continue
                    elif it['pt'] not in title_keys and it['pt'] != item['title'] and len(it['pt'])>3:
                        # print("it['pt']",it['pt'])
                        titles=[]
                        title_keys.append(it)
                        rankclass = classify(model_name_or_path='tkitfiles/rank',num_labels=3,device='cuda')
                        p = rankclass.pre(it['pt'])
                        softmax=rankclass.softmax()
                        one_data={"_id":tt.md5(it['pt']),"title":it['pt'],'key':keyword,'parent':key,'time':time.time(),'rank':p,'softmax':softmax,"state":'uncheck'}
                        rankclass.release()
                        del rankclass
                        print("one_data",one_data)
                        try:
                            DB.pre_titles.insert_one(one_data)
                            pass
                        except:
                            pass
                        
                        # titles.append({"_id":tt.md5(it['pt']),'key':key,"title":it['pt'],'rank':p,'softmax':softmax})
                        titles.append(one_data)
                        print('titles',titles)
                        emit('预测反馈', {'state': 'success','step':'get_titles','data':titles})
                
                            

        i=i+1
        emit('预测反馈', {'state': 'success','step':'log','data':"任务运行结束 "}) 
        # del kws
        # del w2v
                    

@socketio.on('停止预测', namespace='/tapi')
def stop_pre(message):
    print('Client disconnected')
    print('message',message)
    keyword = message.get('data')
    set_var(keyword,{'do':'stop'})

@socketio.on('my event', namespace='/tapi')
def test_message(message):
    # emit('my response', {'data': message['data']})
    titles=[]
    while True:
        time.sleep(5)
        items=DB.pre_titles.find({'key':message['data']})
        titles=[]
        start_time=0
        for item in items:
            print(item)
            titles.append({'title':item['title']})
            start_time=item['time']
            # emit('my response', {'data':item['title']})
        data={"items":titles,"limit":20}
        emit('my response', {'data': data})

# @socketio.on('my broadcast event', namespace='/test')
# def test_message(message):
#     emit('my response', {'data': message['data']}, broadcast=True)

# @socketio.on('connect', namespace='/test')
# def test_connect():
#     emit('my response', {'data': 'Connected'})

@socketio.on('disconnect', namespace='/tapi')
def test_disconnect():
    print('Client disconnected')


if __name__ == "__main__":
    # app.run()
    # socketio.run(app)
    socketio.run(app, host="0.0.0.0", port=5000)