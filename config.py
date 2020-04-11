import sqlite3
import os
import pymongo
from albert_pytorch import classify

from  tkitMarker import  *

#这里定义mongo数据
client = pymongo.MongoClient("localhost", 27017)
DB = client.gpt2Write
print(DB.name)
# DB.my_collection
# Collection(Database(MongoClient('localhost', 27017), u'test'), u'my_collection')
# print(DB.my_collection.insert_one({"x": 10}).inserted_id)

from tkitMarker_bert import Marker
import tkitNextSents

NextS=tkitNextSents.NextSents("tkitfiles/bertNext/")
NextS.load_model()

Pred_Marker=Marker(model_path="./tkitfiles/miaoshu")
Pred_Marker.load_model()





Word2vec_model='/mnt/data/dev/github/w2vec关键词抽取/keyextract_word2vec/model/word2vec_demo.model'
Word2vec_model_WV='/mnt/data/dev/github/w2vec关键词抽取/keyextract_word2vec/model/word2vec_demo.vector.model'
Word2vec_model_save_fast='/mnt/data/dev/github/w2vec关键词抽取/keyextract_word2vec/model/word2vec_demo.vector.fast.model'
def get_p():
    # import tkitFile
    P = Pre()
    P.args['conf'] = "tkitfiles/v0.1/config.json"
    P.args['load_path'] = "tkitfiles/v0.1/pytorch_model.bin"
    P.args['vocab'] = "tkitfiles/v0.1/vocab.txt"
    P.args['label_file'] = "tkitfiles/v0.1/tag.txt"
    P.args['max_length'] = 200
    P.setconfig()
    return P


def get_tner():
    # 初始化提取关系词
    TNer = Pre()
    TNer.args['conf'] = "tkitfiles/ner_rel/config.json"
    TNer.args['load_path'] = "tkitfiles/ner_rel/pytorch_model.bin"
    TNer.args['vocab'] = "tkitfiles/ner_rel/vocab.txt"
    TNer.args['label_file'] = "tkitfiles/ner_rel/tag.txt"
    TNer.args['albert_path'] = "tkitfiles/ner_rel"
    TNer.args['albert_embedding'] = 312
    TNer.args['rnn_hidden'] = 500
    # TNer.args['rnn_hidden'] = 800

    TNer.model_version = 'ner_rel'
    TNer.args['max_length'] = 200
    TNer.setconfig()
    return TNer


def get_ner():
    # 初始化提取实体
    Ner = Pre()
    Ner.args['conf'] = "tkitfiles/ner/config.json"
    Ner.args['load_path'] = "tkitfiles/ner/pytorch_model.bin"
    Ner.args['vocab'] = "tkitfiles/ner/vocab.txt"
    Ner.args['label_file'] = "tkitfiles/ner/tag.txt"
    Ner.args['albert_path'] = "tkitfiles/ner"
    Ner.args['albert_embedding'] = 312
    Ner.args['rnn_hidden'] = 400

    Ner.model_version = 'ner'
    Ner.args['max_length'] = 200
    Ner.setconfig()
    return Ner

def set_temp(key,vaule):
    """
    保存对象
    vaule为对象
    """
    try:
        DB.runvar.insert_one({"_id":key,'value':vaule}) 
    except :
        # del vaule['_id']
        DB.runvar.update_one({'_id':key},   {"$set" :{"_id":key,'value':vaule}}) 
def get_temp(key):
    """
    获取对象
    """
    data=DB.runvar.find_one({'_id':key})
    if data==None:
        return {'_id':key,'value':{}}
    else:
        return data


def set_var(key,vaule):
    """
    保存对象
    vaule为对象
    """
    try:
        DB.runvar.insert_one({"_id":key,'value':vaule}) 
    except :
        # del vaule['_id']
        DB.runvar.update_one({'_id':key},   {"$set" :{"_id":key,'value':vaule}}) 
def get_var(key):
    """
    获取对象
    """
    data=DB.runvar.find_one({'_id':key})
    if data==None:
        return {'_id':key,'value':{}}
    else:
        return data


