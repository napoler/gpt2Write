
from flask import Flask, render_template, request, json, Response, jsonify,escape
from generate import ai,ai_title
from bulid_data import *
import tkitDb,tkitText,tkitFile,tkitNlp
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import gc
import subprocess
import  requests
from pprint import  pprint
from flask_socketio import SocketIO, emit


import tkitMarker,tkitDb,tkitText
import os
from fun import *

from config import *
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


@app.route('/')
def index():
    return render_template("index.html")

@app.route('/edit')
def edit():
    return render_template("edit.html")
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
    text_array={'original':text,
    'items':get_predict(text,plen,n)
    }
    
 
    for x in locals().keys():
        # print("清理函数内存")
        del locals()[x]
    gc.collect()
    return jsonify(text_array)


def get_predict(text,plen,n):
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
            


@app.route("/json/pre/title",methods=['POST'])
@limiter.limit("100/minute, 1/second")    
def json_pre_title():
    """
    构建训练数据
    """
    data = get_post_data()
    print(data)
    text=" [/tt] "+data.get('title')+" [/tt] "+data.get('content')
    text=text[:300]+" [pt]"
    req=ai_title(text=text)
    print(req)
    
    rdata={"items":[],"num":2}
    return jsonify(rdata)












        
    

   

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
    P=tkitMarker.Pre()
    
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


            
# @app.route("/json/search",methods=['GET'])
@socketio.on('预测标题', namespace='/tapi')
def json_search(message):
    """
    构建训练数据
    """
    print('message',message)
    keyword = message.get('data')
    print("关键词",keyword)
    tt=tkitText.Text()
    set_var(keyword,{'do':'start'})

    items=DB.pre_titles.find({'key':keyword})
    titles=[]
    title_keys=[]
    for item in items:
        # print(item)
        titles.append({"title":item['title']})
        if item['title'] not in title_keys:
            title_keys.append(item['title'])
    # data={"items":titles,"limit":20}
    emit('预测反馈', {'state': 'success','step':'get_titles','data':titles})


    # response = requests.get(
    #     'http://0.0.0.0:6801/json/keyword',
    #     params={'keyword': keyword,'limit':4},
    # )
    # http://localhost:6800/schedule.json&project=default&spider=kgbot&setting=DOWNLOAD_DELAY=2&keyword="宠物狗"
    response = requests.post(
        'http://localhost:6800/schedule.json',
        params={'keyword': keyword,'project':'default','spider':'kgbot'},
    )
    if response.status_code ==200:
        print("提交进程",response.json())
        #修改状态为提交搜索成功
        emit('预测反馈', {'state': 'success','step':'add','data':response.json()})
    else:
        emit('预测反馈', {'state': 'fail','step':'add','data':response.json()})
        
    keys=[]
    #循环两次第一次用于预测之前存在的文本
    for i in range(2):
        #对于第一次循环获取文章过少的进行停止让后台爬虫结束，当然也可以去查询
        #这样自定义修整时间
        if i==1 and  len(keys)<4:
            time.sleep(20*i)
        response = requests.get(
            'http://0.0.0.0:6801/json/search',
            params={'keyword': keyword,'limit':20},
        )
        print("获取数据",response.status_code,{'keyword': keyword,'limit':4},)
        if response.status_code ==200:
            # print
            items=response.json()
            emit('预测反馈', {'state': 'success','step':'get_articles','data':items})
            # titles=[]
            print("获取数据数目:",len(items))
            if len(items)>10:
                nsamples=1
            else:
                nsamples=2
            for item in items:
                text=item['content']
                key=tt.md5(text)
                titles=[]
                if key not in keys:
                    keys.append(key)
                    text=text[:300]+" [pt]"
                    pre_title=ai_title(text=text,key=keyword,nsamples=nsamples)
                    for it in pre_title:
                       
                        if it not in title_keys and it != item['title']:
                            title_keys.append(it)
                            titles.append({"title":it})
                    emit('预测反馈', {'state': 'success','step':'get_titles','data':titles})

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
    socketio.run(app)