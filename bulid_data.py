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
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

import json

# from .fun import *


from jieba import analyse
from harvesttext import HarvestText

import pymongo



def add_data(data,path='data/'):
    """
    添加数据样本
    data={"keywords": "哈士奇，主人，嚎叫，便是，说明，思考，没有，犬种，原因，新手，", "content": "新手养狗，哈是无忧无没有机会被惯坏，它是个比较独立自主思考的孩子。\n\n\n独立自己爱思考，加上无忧无虑爱冒险，可以很负责任的说，哈士奇是班级里成绩好，但不听老师管教的学生。\n\n这么说完，很多主人要用亲身体验反驳我了。但现在大多人是这样养哈士奇的：缺乏喂养知识，没有给充足的运动量，看到二哈犯傻只会在旁边哈哈哈，不思考二哈犯二背后的问题。你们养的是哈士奇吗？不，你们养的是二哈啊！\n\n\n\n\n强势的主人\n\n在成为你家的逗比男孩之前，哈士奇可是雪地里的勇士。几个兄弟一起，拉着主人在广袤白茫茫的雪原荒野求生。\n\n\n\n\n经过训练的哈士奇们，在主人的指令之下急速前行。主人必须是能领导控制哈士奇的，不然一旦被丢弃在雪原里，靠两条腿走路生还几率几乎为零。\n\n养二哈，你必须让它知道你是主人，是领导者。当然不是让你用拳头暴力，而是从小就该建立相应的规则，无规矩不成方圆。\n\n都说二哈撒手没，但是如果从小就训练哈士奇随行，撒手没不攻自破。\n\n\n\n\n\n\n充足的运动\n\n一哈顶三虎，三哈沉航母，五哈斗上帝，十哈创世纪 ，百哈毁灭银河系，千哈称霸宇宙第一！俗话说得好，狼若回头，必有缘由。不是报恩，便是报仇。二哈回头，日子到头。不是拆家，便是拆楼。\n网上写出这段话的人，是不是过着袜子日抛，鞋子月抛，沙发半年抛，家具基本年抛的生活？\n\n很多人不了解实质的原因，就给二哈打标签“拆迁队”。二哈的“闹腾”是祖传的，专业的雪橇犬一天可以跑5个马拉松的距离，还能连着跑好几天不休息。\n\n如果不能满足它的运动量，那就不要怪它口不留情了，欲求不满的结果是这样的。\n\n\n\n\n\n\n每日速跑和慢跑几次是非常有必要的，每日3次以上运动，每次20到30分钟，可以根据个体差异调整（但不适用于幼犬）。在运动的过程中，前后肢充足的伸展会让哈士奇形成优美的体态。\n\n\n\n\n\n\nWOOOOOOOOOOOOF\n\n哈士奇这类原始犬种保持了狼的天性，平时很少“汪汪”叫，更多时候是嚎叫。在野外，唱唱山歌可以理解，但在城里，大晚上你还要“大山的子孙哟~~~~~~”，真的很受不了。\n\n\n\n\n二哈其实是不喜欢叫的，嚎叫一定是它想干嘛了。\n\n在陌生环境里，这怂货就会怕得开始嚎；发情的时候，各种深夜嚎叫；想吸引你的注意力也会一直嚎，有些二哈只要你出门没带上它，门一关上就传出狼嚎。\n\n找出原因，对症下药就好。\n\n下面这只哈士奇，主人不让嚎叫，就小声BB来表达不满。\n\n\n\n如果你养了几只哈士奇，仔细听它们嚎叫，每只狗的声调是不一样的，而且一直在变化。这是一种生存策略，即使只有两只哈士奇，这些嚎叫的声音交织在一起，就会给外界一种“有一堆哈士奇”的错觉。\n\n\n\n\n智商低？\n\n谁说哈士奇智商低，信不信我带一只二哈去把你家拆了。\n\n智商这种东西，绝不是一张智商排行表能说明的，况且斯坦利大佬那张表，是按狗狗的易训性和服从性排名的。一纸文凭说明不了很多问题。在冰原上遭遇暴风雪时，世界白茫茫一片，主人根本看不清路。只靠主人带路，一不小心就把自己带沟里了。哈士奇有自己的主意，哪里可以走，哪里有村落，它们自己会判断。\n\n\n\n\n\n\n但现在没得拉雪橇了，越狱、偷吃、跟主人斗智斗勇，才是哈士奇智商的正确使用方式。\n\n\n\n\n\n\n\n\n不适合新手\n\n之前《权游》热播，冰原狼的炫酷导致大家疯狂购买哈士奇，然后又出现了大规模的弃养狂潮。\n\n\n\n\n\n\n很多人没有仔细了解哈士奇的习性，就将它带回家，继而反倒抱怨哈士奇难养？哈士奇并不是善于阿谀和讨好主人的犬种，它们有自己的主意，也有一万个理由不听你的话。\n\n主人好傻，不想理主人；\n\n今天好热，不想理主人；\n\n没有好吃的，不想理主人……\n\n\n\n\n\n\n训练它们是一项很有挑战性的工作，需要主人有足够的经验和耐心。"}

    """
    tkitFile.File().mkdir(path)
    data_path=path+"data.json"
    tjson=tkitFile.Json(file_path=data_path)

    tjson.save(data)
    return   tjson.load()

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
def data_pre_train_file(path='./data/'):
    """
    生成训练样本
    """
    tkit.File().mkdir(path)
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
        data_pre_train_mongo()

if __name__ == '__main__':
    main()
    #执行构建训练样本
    #预先将data/data.json 复制进目录
    # data_pre_train_file('./data/')
    # data_pre_train_file()
    #将data/csv/目录下数据转化为 data.json 需要包含title 和content字段
    # csv_list()
