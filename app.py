
from flask import Flask, render_template, request, json, Response, jsonify,escape
from generate import ai
from bulid_data import *
import tkitDb,tkitText,tkitFile,tkitNlp
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import gc
import subprocess
 
import os
from fun import *

app = Flask(__name__)


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
    if request.content_type.startswith('application/json'):
        data = request.get_data()
        data = json.loads(data)
    else:
        for key, value in request.form.items():
            if key.endswith('[]'):
                data[key[:-2]] = request.form.getlist(key)
            else:
                data[key] = value
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
    tid=str(hash(text))+str(plen)+str(n)
    cmd = "python3 ./generate.py --prefix '''"+text+"''' --length " +str(plen)+" --nsamples "+str(n)+" --tid "+str(tid)
    print("开始处理: "+cmd)
    # print(subprocess.call(cmd, shell=True))
    if subprocess.call(cmd, shell=True)==0:
        data_path="tmp/run_task"+tid+".json"
        print('load',data_path)
        try:
            tjson=tkitDb.Json(file_path=data_path)
            return   tjson.load()[0]['data']
        except:
            print('load文件失败',data_path)
            return{}
            pass
    else:
        return{}
    

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


@app.route("/json/get/keywords",methods=[ 'POST'])
def json_get_keywords():
    """
    构建训练数据
    """
    data= get_post_data()
    ttext=tkitText.Text()
    keywords=ttext.get_keywords(data['text'],num=40)
    return jsonify(keywords)

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


if __name__ == "__main__":
    app.run()