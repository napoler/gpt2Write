
from flask import Flask, render_template, request, json, Response, jsonify,escape
from generate import ai
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
def hello():
    name = request.args.get("name", "World")
    return f'Hello, {escape(name)}!'

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
    text_array={'original':text,
    'items':ai(text,150)
    }
    
 

    return jsonify(text_array)


if __name__ == "__main__":
    app.run()