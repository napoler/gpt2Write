
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import Terry_toolkit as tkit


from pyltp import Parser
from pyltp import SementicRoleLabeller
from pyltp import Segmentor
from pyltp import Postagger
from pyltp import NamedEntityRecognizer
from tqdm import tqdm
from harvesttext import HarvestText
import os
from tkitMarker import *
import tkitText


from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search
from elasticsearch_dsl import Q

def cut_paragraphs(text,num_paras=5):
    tt= tkitText.Text()
    text=tt.sentence_segmentation_v1(text)
    ht0 = HarvestText()
    return ht0.cut_paragraphs("\n".join(text), num_paras)



def search_content(keyword):
    client = Elasticsearch()
    q = Q("multi_match", query=keyword, fields=['title', 'body'])
    # s = s.query(q)

    # def search()
    s = Search(using=client)
    # s = Search(using=client, index="pet-index").query("match", content="金毛")
    s = Search(using=client, index="pet-index").query(q)
    response = s.execute()
    return response
    # for hit in response:
    #     print(hit.meta)
    #     print(hit.meta.score)
    #     print(hit)

def search_sent(keyword):
    client = Elasticsearch()
    q = Q("multi_match", query=keyword, fields=['title', 'content'])
    s = Search(using=client)
    # s = Search(using=client, index="pet-index").query("match", content="金毛")
    s = Search(using=client, index="pet-sent-index").query(q)
    response = s.execute()
    return response


















class Nlp:
    """
    自动标记知识
    """
    def __init__(self,LTP_DATA_DIR = '/mnt/data/dev/model/ltp/ltp_data_v3.4.0' ):

        self.ner_model_path = os.path.join(LTP_DATA_DIR, 'ner.model')  # 命名实体识别模型路径，模型名称为`pos.model`
        self.cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
        self.pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')  # 词性标注模型路径，模型名称为`pos.model`
        self.srl_model_path = os.path.join(LTP_DATA_DIR, 'pisrl.model')  # 语义角色标注模型目录路径，模型目录为`srl`。注意该模型路径是一个目录，而不是一个文件。
        self.par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`
        self.i=0
        # self.tdb=tkit.LDB(path="tdata/lvkg_mark.db")
        # db=tkitDb.LDB(path="/mnt/data/dev/github/标注数据/Bert-BiLSTM-CRF-pytorch/tdata/lvkg.db")

    def ner(self,text):
        """
        获取ｎｅｒ数据
        """
        segmentor = Segmentor()  # 初始化实例
        segmentor.load(self.cws_model_path)  # 加载模型
        words = segmentor.segment(text)  # 分词
        # print ('\t'.join(words))
        segmentor.release()  # 释放模型

        postagger = Postagger() # 初始化实例
        postagger.load(self.pos_model_path)  # 加载模型

        # words = ['元芳', '你', '怎么', '看']  # 分词结果
        postags = postagger.postag(words)  # 词性标注
        # print("##"*30)
        # print ('\t'.join(postags))
        postagger.release()  # 释放模型

        recognizer = NamedEntityRecognizer() # 初始化实例
        recognizer.load(self.ner_model_path)  # 加载模型
        # words = ['元芳', '你', '怎么', '看']
        # postags = ['nh', 'r', 'r', 'v']
        netags = recognizer.recognize(words, postags)  # 命名实体识别
        recognizer.release()  # 释放模型
        words_list=[]
        for word, flag in zip(words, netags):
            # print(word,flag)
            if flag.startswith("B-"):
                one=[]
                one.append(word)
            elif flag.startswith("I-"):
                one.append(word)
            elif flag.startswith("E-"):
                one.append(word)
                words_list.append("".join(one))
            elif flag.startswith("S-"):
                words_list.append(word)
        # print(words_list)
        # return words_list,words, postags,netags
        return words_list





def get_des():
    """
    获取描述
    """
    P = Pre()
    path="tkitfiles/des"
    P.args['conf'] = path+"/config.json"
    P.args['load_path'] = path+"/pytorch_model.bin"
    P.args['vocab'] = path+"/vocab.txt"
    P.args['label_file'] =path+"/tag.txt"
    P.args['max_length'] = 200
    P.model_version = 'des'
    P.setconfig()
    return P


def get_ner_rel():
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
# #获取描述
# MDes=get_des()
# #获取关系词
# MRel=get_ner_rel()
# # 获取实体
# MNer=get_ner()


def get_kg(text):
    """
    预测知识
    """
    nlp=Nlp()
    MNer = get_ner()
    MRel = get_ner_rel()
    MDes = get_des()
    kgs=[]
    result=MNer.pre([text])
    ner=nlp.ner(text)
    # print("ner",ner)
    print(".")
    for item in result[0][1]:
        # print(item)
        if item['type']=="实体":
            ner.append(item['words'])
    
    ner=list(set(ner))
    print("ner",ner)
    for w in ner:
        kg_one=[]
        kg_one.append(w)
        # print(kg_one)
        # print("预测",w+"#"+text)
        result=MRel.pre([w+"#"+text])
        print("预测语句 关系",w+"#"+text)
        # print('关系result',result)
        for item in result[0][1]:
            if item['type']=="关系": 
                kg_one.append(item['words'])
                print('关系',kg_one)
                print("预测语句 描述","#".join(kg_one)+"#"+text)
                result=MDes.pre(["#".join(kg_one)+"#"+text])
                # print('描述',result)
                for item in result[0][1]:
                    if item['type']=="描述":
                        # print(text)
                       
                        kg_one.append(item['words'])
                        print("知识:",kg_one)
                        
                        # print("kg_one",kg_one)
                        kgs.append(kg_one)

    # print(text)
    # print(kgs)


    # pre_re=ner+"#"+text
    # result=TNer.pre([pre_re])
    # # print("预测的关系词::",result)
    # words_list=[]
    # for item in result[0][1]:
    #     if item['type']=="关系":
    #         words_list.append(item['words'])









# def get_keyseq(text,num=20):
#     # LANGUAGE = "chinese"
#     # SENTENCES_COUNT = num
#     # stemmer = Stemmer(LANGUAGE)
#     # summarizer = Summarizer(stemmer)
#     # summarizer.stop_words = get_stop_words(LANGUAGE)
#     ie=tkit.TripleIE(model_path="/mnt/data/dev/model/ltp/ltp_data_v3.4.0")
#     # parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
#     # l=[]
#     # for sentence in summarizer(parser.document, SENTENCES_COUNT):
#     #     l.append(str(sentence))
#     #     # l.append({'text':str(sentence),'with':0})
#     # # del sentence
#     s=[]
#     # print("获取文章重点条数",l)
#     for it in ie.get(text):
#         # print(it)
#         if it==None:
#             pass
#         else:
#             s.append({'text':''.join(list(it)),'with':0})
#     # print(s)
#     # segs_pre.append(' [KW] '+'。'.join(s)+' [SEP] ')
#     return s

def get_sumy(text):
    """
    获取摘要
    """
    LANGUAGE = "chinese"
    SENTENCES_COUNT = 5
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    l=[]
    # print(parser.document)
    items=[]
    # for s in parser.document:
    #         items.append(s)
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        l.append(str(sentence))
    # del sentence 
    # try:
    #     for s in parser.document:
    #         items.append(s)
    # except:
    #     pass

    return l,items
def get_keyseq(text,num=20):
    """
    获取关键内容
    三元组抽取    
    """
    ht=HarvestText()
    s=[]
    text=tkit.Text().clear(text)
    for item in ht.triple_extraction(sent=text, standard_name=False, stopwords=None, expand = "all"):
        if item=='':
            pass
        else:
            print(' '.join(item))
            # s.append(str(item))
            s.append({'text':''.join(item),'with':0})
    # s="。".join(s)


    return s