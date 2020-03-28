import tkitText
from pprint import pprint
from albert_pytorch import *


class NextSent:
    def __init__(self):
        pass
    def __del__(self):
        self.model.release()
    def load(self):
        self.model = classify(model_name_or_path='../tkitfiles/next', num_labels=2, device='cuda')
        pass
    def takeSecond(self,elem):
        return elem[1]
    def pre(self,text_a,sents):
        # tt=tkitText.Text()
        # sents=tt.sentence_segmentation_v1(text_b)
        li=[]
        for i,sent in enumerate( sents):
            p = self.model.pre(text_a,sent)
            next_rank=self.model.softmax()[1]
            li.append((sent,next_rank,i))
            # print(next_rank,sent)
            # rankclass.release()
        li.sort(key=self.takeSecond,reverse=True)

        # pprint(li)

        return li
    def bulid_text(self,li):
        li=li[1:]
        text_li=[]
        for it in li:
            text_li.append(it[0])
        return text_li
    def auto_pre_one(self,start,text):
        tt=tkitText.Text()
        sents=tt.sentence_segmentation_v1(text)
        text_a=start
        li=self.pre(text_a[-200:],sents)

        return li
    def auto_pre(self,start,text,limit=0.7):
        tt=tkitText.Text()
        sents=tt.sentence_segmentation_v1(text)

        for i in range(len(sents)):
            if i==0:
                text_a=start
                print(start)
            else:
                text_a=text_a+li[0][0]
                print(i,li[0][1],li[0][0])
            # text_a=text_a[-200:]

            li=self.pre(text_a[-200:],sents)
            sents=self.bulid_text(li)
            if li[0][1]<=limit:
                break
        # print(text_a)
        return text_a
    def auto_sort(self,start,text,limit=0.7):
        """
        对内容自动排序
        """
        return self.auto_pre(start,text,limit)

