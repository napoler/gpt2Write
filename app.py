
from flask import Flask, render_template, request, json, Response, jsonify,escape
from generate import ai
from bulid_data import *
import Terry_toolkit as tkit
app = Flask(__name__)

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
    'items':ai(text,plen,n)
    }
    
 

    return jsonify(text_array)


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
    ttext=tkit.Text()
    keywords=ttext.get_keywords(data['text'],num=10)
    return jsonify(keywords)



if __name__ == "__main__":
    app.run()