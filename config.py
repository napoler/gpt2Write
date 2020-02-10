import sqlite3
import os
import pymongo


#这里定义mongo数据
client = pymongo.MongoClient("localhost", 27017)
DB = client.gpt2Write
print(DB.name)
# DB.my_collection
# Collection(Database(MongoClient('localhost', 27017), u'test'), u'my_collection')
# print(DB.my_collection.insert_one({"x": 10}).inserted_id)


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


