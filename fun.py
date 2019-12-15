
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

from harvesttext import HarvestText


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