# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
# import Terry_toolkit as tkit
import tkitDb,tkitText,tkitFile,tkitNlp

import os
# data=[{ "keywords": "学习,学校","title": "借鉴：这篇最受欢迎校训，没有一个字讲学习",  "content": "“我知道，我不是因为偶然才来到这个世界，我是为"}]
from MagicBaidu import MagicBaidu
import pprint
import numpy as np
import csv
from tqdm import tqdm
import re
import argparse

from config import  *

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer

from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import json
import  tkitText
# from .fun import *
import tkitW2vec

from jieba import analyse
from harvesttext import HarvestText

import pymongo
# import  macropodus
import tiktThreading
import jieba
def add_data(data,path='data/'):
    """
    添加数据样本
    data=[{"keywords": "哈士奇，主人，嚎叫，便是，说明，思考，没有，犬种，原因，新手，", "content": "新手养狗，哈是无忧无的经验和耐心。"}]

    """
    tkitFile.File().mkdir(path)
    data_path=path+"data.json"
    tjson=tkitFile.Json(file_path=data_path)

    tjson.save(data)
    return   tjson.auto_load()

# def search(keywords=''):
#     tsearch=tkit.SearchBaidu()
#     # articles=tsearch.get_full(keyword=keywords)
#     # print(articles)
#     ls=  tsearch.get(keyword=keywords,start=0)


#     return ls
def get_seq(text):
    """
    获取关键内容
    三元组抽取    
    """
    ht=HarvestText()
    s=[]
    text=tkitText.Text().clear(text)
    for item in ht.triple_extraction(sent=text, standard_name=False, stopwords=None, expand = "all"):
        if item=='':
            pass
        else:
            # print(' '.join(item))
            # s.append(str(item))
            s.append(''.join(item))
    # s="。".join(s)


    return s
from memory_profiler import profile
import gc
# @profile
def data_pre_train( data_path='data/data.json',train_path='data/train.txt' ):
    """
    from=0  #文章开始id
    limit=10 # 返回文章数目
    >>>data_pre_train(from=0, limit=10)
    [unused5] 标记关键词
      [unused6]  标记标题
    [unused7]  标记前文标题
       [unused8]  标记正文
    """
    LANGUAGE = "chinese"
    SENTENCES_COUNT = 10
    article_max_len=500
    # tjson=tkit.Json(file_path=data_path)
    # data=tjson.auto_load()
    # print(len(data))
    ttext=tkitText.Text()
    # extractor = tkit.TripleExtractor()
    # if len(data)>tfrom+limit:
    #     data=data[tfrom:tfrom+limit]
    # elif len(data)<tfrom:
    #     print("数据过短了，存在问t")
    #     return []
    # else:
    #     data=data[tfrom:]
    # for item in tjson.auto_load():
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    ie=tkitNlp.TripleIE(model_path="/mnt/data/dev/model/ltp/ltp_data_v3.4.0")
    f1 = open(train_path,'w')
    articles=[]
    # 引入TF-IDF关键词抽取接口
    tfidf = analyse.extract_tags
    # 引入TextRank关键词抽取接口
    textrank = analyse.textrank
    with open(data_path, 'r', encoding = 'utf-8') as data:
        for art_i,it in tqdm(enumerate(data)):
            item=json.loads(it[:-1])
            # if art_i%10==0:
            #     print('arti', art_i)
            segs_pre=[]
            segs_end=[]
            # # segs_pre.append(' [KW] '+item['keywords']+' [SEP] ')
            # # l=ttext.summary( item['content'],num=10)
            # # extractor = tkit.TripleExtractor()
            # # svos = extractor.triples_main(item['content'])

            # # extractor.clear()
            # # print('svos', svos)
            # parser = PlaintextParser.from_string(item['content'], Tokenizer(LANGUAGE))
            # l=[]
            # for sentence in summarizer(parser.document, SENTENCES_COUNT):
            #     l.append(str(sentence))
            # # del sentence
            s=[]      




            # # 这里开始处理关键词 关键语句等信息 
            # try:
            #     for it in ie.get(item['title']+'\n'+item['content']):
            #         # print(it)
            #         if it==None:
            #             pass
            #         else:
            #             s.append(''.join(list(it)))
            #     # print(s)
            # except:
            #     pass
            # # s=get_seq(item['title']+'\n'+item['content'])
            # # 基于TextRank算法进行关键词抽取
            keywords = textrank(item['title']+'\n'+item['content'], topK=10, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')) 
            # # 输出抽取出的关键词
            # # print(keywords)
            # # for keyword in keywords:
            # #     print (keyword + "/",)
            # # 基于TF-IDF算法进行关键词抽取
            # # keywords = tfidf(item['title']+'\n'+item['content'], topK=10, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))
            # # print(keywords)
            # # 输出抽取出的关键词
            # # for keyword in keywords:
            # #     print( keyword + "/",)
            # # keywords1 =ttext.get_keywords(item['title']+'\n'+item['content'])
            # # new_keywords=[]
            # # for keyword in keywords1:
            # #     new_keywords.append(keyword['word'])        
            # # keywords =ttext.get_keyphrases(item['title']+'\n'+item['content'])
            # # kws=keywords+new_keywords
            # # # s.append('，'.join(kws))
            # s=['，'.join(keywords)]+s
            segs_pre.append(' [KW] '+'，'.join(keywords)+' [/KW] ')
            # del s
            # # svos = extractor.triples_main('。'.join(l))
            # #这里的屏蔽内容









            try:
                segs_pre.append(' [TT] '+item['title']+" [/TT] ")
                segs_end.append(' [PT] '+item['title']+" [/PT] ")
            except:
                pass
            segs=sentence_seg(" [CLS] "+item['content']+" [END] ")
            article="".join(segs_pre+segs+segs_end)
            
            one=[]
            for i in range(len(article)//article_max_len+1):
                #截取内容
                one.append(article[i*article_max_len:(i+1)*article_max_len]+"")
            articles.append("\n".join(one)+"")
            if art_i%100==0:
                print('arti', art_i)
                # f1.write("\n\n".join(articles)+"\n\n")
                f1.write("\n\n".join(articles)+"")
                articles=[]
            # del articles
            del segs
        f1.write("\n\n".join(articles)+"")
        f1.close()
        gc.collect()
        del stemmer
        del summarizer
        del ie


        gc.collect()
        return
def cut_text(text,lenth):
    """
    分割固定长度字符串
    """
    textArr = re.findall('.{'+str(lenth)+'}', text)
    textArr.append(text[(len(textArr)*lenth):])
    return textArr

def check_one_Process(key):
    client = pymongo.MongoClient("localhost", 27017)
    DB_kg_scrapy = client.kg_scrapy
    q={'_id':key}
    # print(DB_kg_scrapy.kg_content_Processed.find_one(q))
    return DB_kg_scrapy.kg_content_Processed.find_one(q)

def add_one_Process(item):
    client = pymongo.MongoClient("localhost", 27017)
    DB_kg_scrapy = client.kg_scrapy
    try:
        DB_kg_scrapy.kg_content_Processed.insert_one(item)
    except :
        pass
def data_pre_train_mongo_Process( data_path='data/data.json',train_path='data/train_db_kwseq.txt' ):
    """
    对抓取的数据进行预处理
    生成摘要和关键词
    """
    LANGUAGE = "chinese"
    SENTENCES_COUNT = 5
    # w2vWV=tkitW2vec.Word2vec()
    # w2vWV.load(model_file=Word2vec_model)

    # tt=tkitText.Text()
    # stemmer = Stemmer(LANGUAGE)
    # summarizer = Summarizer(stemmer)
    # summarizer.stop_words = get_stop_words(LANGUAGE)
    jieba.load_userdict('dict.txt')
    jieba.analyse.set_stop_words('stopwords.txt')
    textrank = jieba.analyse.textrank
    #这里定义mongo数据
    client = pymongo.MongoClient("localhost", 27017)
    DB_kg_scrapy = client.kg_scrapy
    q={}
    i=0
    for item in tqdm(DB_kg_scrapy.kg_content.find(q)):
        i=i+1
        if check_one_Process(item['_id']):
            # print("已存在")
            pass
        else:
 
            if len(item['content'])>500:
                SENTENCES_COUNT = 5
            else:
                SENTENCES_COUNT = 3
            sm=[]
            try:
                sm=w2v.summary(item['content'],topn=SENTENCES_COUNT)
            except :
                pass
            keywords=[]
            try:
                # # # 基于TextRank算法进行关键词抽取
                keywords = textrank(item['title']+'\n'+item['content'], topK=10, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))      
                keyphrases =tt.get_keyphrases(item['title']+'\n'+item['content'])
                keywords=keywords+keyphrases
                keywords=list(set(keywords))
                # kws=w2vWV.keywords(item['title']+"　"+item['content'])
                # for word,rank in kws[:20]:
                #     keywords.append(word)
            except :
                pass
            # print(keywords)
            item['summary']=sm
            item['keywords']=keywords
            try:
                add_one_Process(item)
            except:
                pass
 
def data_pre_train_mongo( data_path='data/data.json',train_path='data/train_db.txt' ):
    """
    from=0  #文章开始id
    limit=10 # 返回文章数目
    >>>data_pre_train(from=0, limit=10)
    [unused5] 标记关键词
      [unused6]  标记标题
    [unused7]  标记前文标题
       [unused8]  标记正文
    """
    LANGUAGE = "chinese"
    SENTENCES_COUNT = 10
    article_max_len=500
    ttext=tkitText.Text()


    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    # ie=tkitNlp.TripleIE(model_path="/mnt/data/dev/model/ltp/ltp_data_v3.4.0")
    f1 = open(train_path,'w')
    articles=[]
    # 引入TF-IDF关键词抽取接口
    tfidf = analyse.extract_tags
    # 引入TextRank关键词抽取接口
    textrank = analyse.textrank
    #这里定义mongo数据
    client = pymongo.MongoClient("localhost", 27017)
    DB_kg_scrapy = client.kg_scrapy
    print(DB.name)
    q={}
    # print('q',q)
    for item in DB_kg_scrapy.kg_content.find(q):
        # print(item)
        content=" [TT] "+ item['title']+" [/TT]  "+item['content']+" [PT] "+ item['title']+" [/PT] [END]"
        content=content.replace("\n\n\n", "\n\n")
        content=content.replace("\n", " [SEP] ")
        content_list=cut_text(content,480)
        # for it in content_list:
        #     print("++++"*20)
        #     print(it)
        # f1.write("\n".join(content_list)+"")
        f1.write(content)
        f1.write("\n")
        #     print(len(it))


    # with open(data_path, 'r', encoding = 'utf-8') as data:
    #     for art_i,it in tqdm(enumerate(data)):
    #         item=json.loads(it[:-1])
    #         # if art_i%10==0:
    #         #     print('arti', art_i)
    #         segs_pre=[]
    #         segs_end=[]

    #         s=[]      

    #         keywords = textrank(item['title']+'\n'+item['content'], topK=10, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')) 
      
    #         segs_pre.append(' [KW] '+'，'.join(keywords)+' [/KW] ')

    #         try:
    #             segs_pre.append(' [TT] '+item['title']+" [/TT] ")
    #             segs_end.append(' [PT] '+item['title']+" [/PT] ")
    #         except:
    #             pass
    #         segs=sentence_seg(" [CLS] "+item['content']+" [END] ")
    #         article="".join(segs_pre+segs+segs_end)
            
    #         one=[]
    #         for i in range(len(article)//article_max_len+1):
    #             #截取内容
    #             one.append(article[i*article_max_len:(i+1)*article_max_len]+"")
    #         articles.append("\n".join(one)+"")
    #         if art_i%100==0:
    #             print('arti', art_i)
    #             # f1.write("\n\n".join(articles)+"\n\n")
    #             f1.write("\n\n".join(articles)+"")
    #             articles=[]
    #         # del articles
    #         del segs
    #     f1.write("\n\n".join(articles)+"")
    #     f1.close()
    #     gc.collect()
    #     del stemmer
    #     del summarizer
    #     del ie


    #     gc.collect()
    #     return


        # articles_num=art_i+1
        # return articles,articles_num

def data_pre_train_mongo_kwseq( data_path='data/data.json',train_path='data/train_db_kwseq.txt' ):
    """
    from=0  #文章开始id
    limit=10 # 返回文章数目
    >>>data_pre_train(from=0, limit=10)
    [unused5] 标记关键词
      [unused6]  标记标题
    [unused7]  标记前文标题
       [unused8]  标记正文
    """
    LANGUAGE = "chinese"
    SENTENCES_COUNT = 10
    article_max_len=500


    tt=tkitText.Text()
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    # ie=tkitNlp.TripleIE(model_path="/mnt/data/dev/model/ltp/ltp_data_v3.4.0")
    f1 = open(train_path,'w')
    articles=[]
    # 引入TF-IDF关键词抽取接口
    # tfidf = analyse.extract_tags
    jieba.load_userdict('dict.txt')
    jieba.analyse.set_stop_words('stopwords.txt')
    textrank = jieba.analyse.textrank
    #这里定义mongo数据
    client = pymongo.MongoClient("localhost", 27017)
    DB_kg_scrapy = client.kg_scrapy
    print(DB.name)
    q={}
    # print('q',q)
    # w2v=tkitW2vec.Word2vec()
    # w2v.load(model_file=Word2vec_model)
    i=0
    for item in DB_kg_scrapy.kg_content.find(q):
        i=i+1
        if i==100:
            break
        # print(item)
        # for sent in tt.sentence_segmentation_v1(item['content']):
        #     print("句子:",sent)
        #     keywords=[]
        #     # print(w2v.keywords([sent]))
        #     try:
        #         for word,rank in w2v.keywords(sent):
        #             # print(word,"----》",rank )
        #             keywords.append(word)
        #     except :
        #         pass
        #     print(keywords)
        # keywords=[]
        #     # print(w2v.keywords([sent]))
        # try:
        #     for word,rank in w2v.keywords(item['content'])[:20]:
        #         # print(word,"----》",rank )
        #         keywords.append(word)
        # except :
        #     pass

        # print('keywords1',keywords)
        try:
            keywords = textrank(item['content'], topK=10, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v')) 
            print('keywords2',keywords)
            content="[KW] "+" ".join(keywords)+" [/KW] [TT] "+ item['title']+" [/TT]  "+item['content']+" [PT] "+ item['title']+" [/PT] [END]"
            content=content.replace("\n\n\n", "\n\n")
            content=content.replace("\n", " [SEP] ")
            content_list=cut_text(content,480)
            # for it in content_list:
            #     print("++++"*20)
            #     print(it)
            # f1.write("\n".join(content_list)+"")
            f1.write(content)
            f1.write("\n")
        except :
            pass
from harvesttext import HarvestText


def get_one():
    client = pymongo.MongoClient("localhost", 27017)
    DB_kg_scrapy = client.kg_scrapy
    q={}
    for item in DB_kg_scrapy.kg_content.find(q):
        # data.append(it)
        yield  item
        # 
LANGUAGE = "chinese"
SENTENCES_COUNT = 5
# article_max_len=500
tt=tkitText.Text()
stemmer = Stemmer(LANGUAGE)
summarizer = Summarizer(stemmer)
summarizer.stop_words = get_stop_words(LANGUAGE)

jieba.load_userdict('dict.txt')
jieba.analyse.set_stop_words('stopwords.txt')
textrank = jieba.analyse.textrank

w2v=tkitW2vec.Word2vec()
w2v.load(model_file=Word2vec_model)
def add_one(args):
# def add_one(args,se):
    # se.acquire()
    item=args['item']
    f1=args['f1']

    if len(item['content'])>500:
        SENTENCES_COUNT = 5
    else:
        SENTENCES_COUNT = 3
    sm=[]
    # try:
    #     if len(item['content'])>500:
    #         SENTENCES_COUNT = 10
    #     else:
    #         SENTENCES_COUNT = 5
    #     parser = PlaintextParser.from_string(item['content'], Tokenizer(LANGUAGE))
    #     for sentence in summarizer(parser.document, SENTENCES_COUNT):
    #         sm.append(str(sentence))
    # except :
    #     pass
    # 文本摘要(summarization, 可定义方法, 提供9种文本摘要方法, 'lda', 'mmr', 'textrank', 'text_teaser')
    # sm=[]
    # try:
    #     summary= item['content']
    #     sents = macropodus.summarization(text=str(summary),num=5, type_summarize="text_teaser",title=str(item['title']))
    

    #     for r,s in sents:
    #         sm.append(s)
    # except :
    #     pass
    try:
        sm=w2v.summary(item['content'],topn=SENTENCES_COUNT)
    except :
        pass
 
    
    keywords=[]
    try:
        # # 基于TextRank算法进行关键词抽取
        keywords = textrank(item['title']+'\n'+item['content'], topK=10, withWeight=False, allowPOS=('ns', 'n', 'vn', 'v'))  
        # kws=w2v.keywords(item['title']+'\n'+item['content'])
        # for word,rank in kws:
        #     keywords.append(word)
        # keywords1 =tt.get_keywords(item['title']+'\n'+item['content'])
        # print(keywords1)
        # new_keywords=[]
        # for keyword in keywords1:
        #     new_keywords.append(keyword['word'])        
        keyphrases =tt.get_keyphrases(item['title']+'\n'+item['content'])
        # print(new_keywords)
        keywords=keywords+keyphrases
        keywords=list(set(keywords))
    except :
        pass

    # print(keywords)
    content=" [SM] "+"".join(sm)+" [/SM] [KW] "+" ".join(keywords)+" [/KW]  [TT] "+ item['title']+" [/TT]  "+item['content']+" [PT] "+ item['title']+" [/PT] [END]"
    # content=" [KW] "+" ".join(keywords)+" [/KW]  [TT] "+ item['title']+" [/TT]  "+item['content']+" [PT] "+ item['title']+" [/PT] [END]"
    content=content.replace("\n\n\n", "\n\n")

    content=content.replace("\n", " [SEP] ")

    # content_list=cut_text(content,480)
    # for it in content_list:
    #     print("++++"*20)
    #     print(it)
    # f1.write("\n".join(content_list)+"")
    f1.write(content)
    f1.write("\n")
    # se.release()
def data_pre_train_mongo_summary( data_path='data/data.json',train_path='data/train_db_Summary.txt' ):
    """
    from=0  #文章开始id
    limit=10 # 返回文章数目
    >>>data_pre_train(from=0, limit=10)
 
 
    """

    f1 = open(train_path,'w')

    i=0
    # data=[]
    # tt=tiktThreading.TT(5)
    # for item in get_one():
    tjson=tkitFile.Json(file_path=data_path)
    for item in tqdm(tjson.auto_load()):
        i=i+1
        if i%10000==0:
            print(i)
        args={'item':item,'f1':f1}
        add_one(args)
        # tt.load(add_one,args)
        # tt.start()
def data_pre_train_mongo_to_json( data_path='data/data.json',train_path='data/train_db_Summary.txt' ):
    """
    from=0  #文章开始id
    limit=10 # 返回文章数目
    >>>data_pre_train(from=0, limit=10)
 
 
    """

    i=0
    data=[]
    for item in get_one():
        i=i+1
        data.append(item)
        if i%10000==0:
            print(i)
            try:
                add_data(data)
                data=[]
            except :
                pass
    add_data(data)
            
        
def data_pre_train_file(path='./data/'):
    """
    生成训练样本
    """
    tkitFile.File().mkdir(path)
    train_path=path+'train.txt'
    task_path=path+'task.json'
    data_path=path+'data.json'
    tjson=tkitFile.Json(file_path=task_path)

    # try:
    #     tasks=tjson.load()
    #     task=tasks[0]
    #     os.remove(task_path)
    # except:
    #     # task=[]
    #     task={"tfrom":0,'limit':10}
    data_pre_train(data_path=data_path,train_path=train_path)


    # f1.write('hello boy!')
    # articles,articles_num=data_pre_train(tfrom=task['tfrom'], limit=task['limit'], data_path=data_path)
    # if len(articles)>0:
    #     f1 = open(train_path,'w')
    #     # print(articles)
    #     f1.write("\n".join(articles))
    #     f1.close()
    #     # task['tfrom']=task['tfrom']+articles_num
    #     # tjson.save([task])
    #     train_info={
    #         'task':task,
    #         'path':task_path
    #     }
    #     return train_info
    # else:
    #     return []

from textrank4zh import TextRank4Keyword, TextRank4Sentence
def  sentence_seg(text):
    segs=tkitText.Text().sentence_segmentation_v1(text)
    return segs
def csv_list(path="data/csv/"):
    f = tkitFile.File()
    csv_list=f.file_List(path, type='csv')
    for line in csv_list:
        print('add:',line)
        try:
            data=csv_data(file_path=line)

            add_data(data=data)
        except:
            print('csv文件有误跳过')




def csv_data(file_path=''):
    d=tkitFile.Csv().csv_data(file_path=file_path)
    ttext=tkitText.Text()
    # print(d[10])
    new_data=[]
    for item in tqdm(d):
        # print(item)
        if item['title'] == '' or item['content'] == '':
            # print(",哦哦哦")
            pass
        else:
            kwords=ttext.get_keywords(item['title']+' '+item['content'],num=40)
            keywords=[]
            for it in kwords:
                keywords.append(it['word'])
            # keywords=keywords
            data_one={'keywords':'，'.join(keywords),'title':item['title'],'content':item['content']}
            yield data_one
    #         new_data.append(data_one)
    # return new_data
def main():
    parser = argparse.ArgumentParser(usage="运行数据构建.", description="help info.")
    parser.add_argument("--do", type=str, default='data_pre_train_file',required=False, help="输入运行的类型  (csv_list（将csv文件转换成data.json）,data_pre_train_file(构建豫训练) )")
    args = parser.parse_args()
    if args.do == 'csv_list':
        csv_list()
    elif args.do == 'data_pre_train_file':
        data_pre_train_file('./data/')
    elif  args.do=='data_pre_train_mongo':
        #生成训练资料
        # python bulid_data.py --do data_pre_train_mongo
        data_pre_train_mongo()
    elif  args.do=='data_pre_train_mongo_kwseq':
        #生成训练资料
        # python bulid_data.py --do data_pre_train_mongo_kwseq
        data_pre_train_mongo_kwseq()
    elif  args.do=='data_pre_train_mongo_summary':
        #生成训练资料
        # python bulid_data.py --do data_pre_train_mongo_summary
        data_pre_train_mongo_summary()
    elif  args.do=='data_pre_train_mongo_to_json':
        #导出为json
        # python bulid_data.py --do data_pre_train_mongo_to_json
        data_pre_train_mongo_to_json()
    elif  args.do=='data_pre_train_mongo_Process':
        #导出为json
        # python bulid_data.py --do data_pre_train_mongo_Process
        data_pre_train_mongo_Process()

if __name__ == '__main__':
    main()
    #执行构建训练样本
    #预先将data/data.json 复制进目录
    # data_pre_train_file('./data/')
    # data_pre_train_file()
    #将data/csv/目录下数据转化为 data.json 需要包含title 和content字段
    # csv_list()
