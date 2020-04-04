import tkitText
from pprint import pprint
from albert_pytorch import *
from harvesttext import HarvestText

class NextSent:
    def __init__(self):
        pass
    def __del__(self):
        self.model.release()
    def load(self):
        self.model = classify(model_name_or_path='./tkitfiles/next', num_labels=2, device='cuda')
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
        # li.sort(key=self.takeSecond,reverse=False)
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
        # sents=tt.sentence_segmentation_v1(text)
        ht0 = HarvestText()
        sents=ht0.cut_paragraphs(text, 50)

        text_a=start
        li=self.pre(text_a[-200:],sents)

        return li
    # def auto_pre_one_sents(self,start,sents):
        
    def auto_pre(self,start,text,limit=0.5):
        tt=tkitText.Text()
        sents=tt.sentence_segmentation_v1(text)
        text_a=start
        for i in range(len(sents)):
            if i==0:
                text_a=start
                # print(start)
            else:
                text_a=text_a+"\n"+li[0][0]
            li=self.pre(text_a[-200:],sents)
            # print("li",li)
            sents=self.bulid_text(li)
            # print("text_a",text_a)

            if li[0][1]<=limit:
                print(li[0][1])
                break
            yield text_a
        # print(text_a)
        # return text_a
    def auto_sort(self,start,text,limit=0.7):
        """
        对内容自动排序
        """
        # print(start,text)
        # items=[]
        # for li in self.auto_pre(start,text,limit):
        #     items.append(li)
        # return items
        return  self.auto_pre(start,text,limit)