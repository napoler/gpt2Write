import torch
import torch.nn.functional as F
import os,sys
import argparse
from tqdm import trange
from pytorch_transformers import GPT2LMHeadModel
import gc
import Terry_toolkit as tkit
from memory_profiler import profile
import  tkitText
from config import  *
import subprocess
from pprint import pprint
import weakref
from copy import copy 
import tiktThreading
import time
import queue
import uuid 
import gc
gc.set_threshold(700, 10, 5)
# import resource
def is_word(word):
    for item in list(word):
        if item not in 'qwertyuiopasdfghjklzxcvbnm':
            return False
    return True


def _is_chinese_char(char):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    cp = ord(char)
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
            (cp >= 0x3400 and cp <= 0x4DBF) or  #
            (cp >= 0x20000 and cp <= 0x2A6DF) or  #
            (cp >= 0x2A700 and cp <= 0x2B73F) or  #
            (cp >= 0x2B740 and cp <= 0x2B81F) or  #
            (cp >= 0x2B820 and cp <= 0x2CEAF) or
            (cp >= 0xF900 and cp <= 0xFAFF) or  #
            (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits




# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
#     parser.add_argument('--length', default=-1, type=int, required=False, help='生成长度')
#     parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
#     parser.add_argument('--nsamples', default=10, type=int, required=False, help='生成几个样本')
#     parser.add_argument('--temperature', default=1, type=float, required=False, help='生成温度')
#     parser.add_argument('--topk', default=8, type=int, required=False, help='最高几选一')
#     parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
#     parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
#                         help='模型参数')
#     parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='词表路径')
#     parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
#     parser.add_argument('--prefix', default='哈士奇', type=str, required=False, help='生成文章的开头')
#     parser.add_argument('--remove_prefix', default=True, required=False, help='生成文章的开头')
#     parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
#     parser.add_argument('--segment', action='store_true', help='中文以词为单位')
#     parser.add_argument('--fast_pattern', action='store_true', help='采用更加快的方式生成文本')
#     parser.add_argument('--save_samples', action='store_true', help='保存产生的样本')
#     parser.add_argument('--save_samples_path', default='.', type=str, required=False, help="保存样本的路径")

#     args = parser.parse_args()
#     print('args:\n' + args.__repr__())

#     if args.no_wordpiece:
#         from tokenizations import tokenization_bert_without_wordpiece as tokenization_bert
#     elif args.segment:
#         from tokenizations import tokenization_bert_word_level as tokenization_bert
#     else:
#         from tokenizations import tokenization_bert

#     os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
#     length = args.length
#     batch_size = args.batch_size
#     nsamples = args.nsamples
#     temperature = args.temperature
#     topk = args.topk
#     topp = args.topp

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
#     model = GPT2LMHeadModel.from_pretrained(args.model_path)
#     model.to(device)
#     model.eval()

#     if length == -1:
#         length = model.config.n_ctx - len(args.prefix)
#     elif length > model.config.n_ctx - len(args.prefix):
#         raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)
#     if args.save_samples:
#         if not os.path.exists(args.save_samples_path):
#             os.makedirs(args.save_samples_path)
#         samples_file = open(args.save_samples_path + '/samples.txt', 'w', encoding='utf8')
#     while True:
#         raw_text = args.prefix
#         context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
#         generated = 0
#         end=[]
#         for _ in range(nsamples // batch_size):
#             out = generate(
#                 model=model,
#                 context=context_tokens,
#                 end=end,
#                 length=length,
#                 is_fast_pattern=args.fast_pattern,
#                 temperature=temperature, top_k=topk, top_p=topp, device=device
#             )
#             for i in range(batch_size):
#                 generated += 1
#                 text = tokenizer.convert_ids_to_tokens(out)
#                 for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
#                     if is_word(item) and is_word(text[i + 1]):
#                         text[i] = item + ' '
#                 for i, item in enumerate(text):
#                     if item == '[MASK]':
#                         text[i] = ''
#                     if item == '[CLS]' or item == '[SEP]':
#                         text[i] = '\n'
#                 info = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n"
#                 # print(info)
#                 text = ''.join(text).replace('##', '').strip()
#                 # print(text)
#                 if args.remove_prefix:
#                     remove_prefix_length =len(args.prefix)
#                     text=text[remove_prefix_length:]

                
                
#                 if args.save_samples:
#                     samples_file.write(info)
#                     samples_file.write(text)
#                     samples_file.write('\n')
#                     samples_file.write('=' * 90)
#                     samples_file.write('\n' * 2)
#         print("=" * 80)
#         if generated == nsamples:
#             # close file when finish writing.
#             if args.save_samples:
#                 samples_file.close()
#             break

def dump_garbage():
    """ show us what the garbage is about """
    # Force collection
    print( "\nGARBAGE:")
    gc.collect()
    print ("\nGARBAGE OBJECTS:")
    for x in gc.garbage:
        print( 'hello')
        s = str(x)
        #if len(s) > 80: s = s[:77]+'...'
        print( type(x),"\n  ", s)

# ————————————————
# 版权声明：本文为CSDN博主「小熊_晶晶」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/xiongaijing/article/details/12857207
from tokenizations import tokenization_bert
# device = "cuda" if torch.cuda.is_available() else "cpu"

# tokenizer = tokenization_bert.BertTokenizer(vocab_file='model/mini/vocab.txt')
# model = GPT2LMHeadModel.from_pretrained('model/mini/')
# model.to(device)
# model.eval()

class Ai:
    def __init__(self):
        pass
    def load_model(self,path='model/mini/'):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        tokenizer = tokenization_bert.BertTokenizer(vocab_file=path+'vocab.txt')
        model = GPT2LMHeadModel.from_pretrained(path)
        model.to(device)
        model.eval()
        return model, tokenizer
    def __del__(self):
        print("Ai结束")
        # self.release()
    # @profile
    def release(self):
        print("Ai结束")
        model.cpu()
        torch.cuda.empty_cache()
        # try:
        #     del self.model
        #     del self.tokenizer
        # except :
        #     pass
        # del out
        # for x in locals().keys():
        #     print("清理函数内存",locals()[x])
        #     # del locals()[x]
        # gc.collect()
        gc.collect()

    def sample_sequence(self,model, context,end, length, temperature=1, top_k=0, top_p=0.0, device='cpu'):
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0)
        end = torch.tensor(end, dtype=torch.long, device=device)
        generated = context
        generated_out=torch.tensor([], dtype=torch.long, device=device)
        state="No"
        with torch.no_grad():
            for _ in trange(length):
                # print('generated',generated)
                if len(generated[0])==1000:
                    # print(len(generated[0]))
                    generated=generated[:,-950:]
                else:
                    # print(len(generated[0]))
                    pass

                inputs = {'input_ids': generated}
                outputs = model(
                    **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                next_token_logits = outputs[0][0, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                # print("next_token.unsqueeze(0)",end,next_token.unsqueeze(0)[0])
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
                generated_out= torch.cat((generated_out, next_token.unsqueeze(0)), dim=1)

                #设置自动停止
                if next_token.unsqueeze(0)[0] in end :
                    state="Yes"
                    break
        # return generated.tolist()[0]
        return generated_out.tolist()[0],state

    # @profile
    def fast_sample_sequence(self,model, context,end, length, temperature=1, top_k=0, top_p=0.0, device='cpu'):
        # print("devicedevice",device)
        
        inputs = torch.LongTensor(context).view(1, -1).to(device)
        # print(inputs)
        if len(context) > 1:
            # print("gc num",gc.get_count())
            _, past_pre = model(inputs[:, :-1], None)[:2]
            # print("gc num",gc.get_count())
            past=copy(past_pre)
            del past_pre
            del _
            gc.collect()
            # for x in gc.garbage:
            #     # print("x",x)
            #     pass
            _=None
            # model.cpu()
            del _
            gc.collect()
            prev = inputs[:, -1].view(1, -1).to(device)
        else:
            past = None
            prev = inputs
        generate = [] + context
        generated_out=[]
        state="No"
        with torch.no_grad():
            for i in trange(length):
                # print("prev",prev)
                # print("past",len(past))
                output = model(prev, past=past)
                output, past = output[:2]
                output = output[-1].squeeze(0) / temperature
                filtered_logits = top_k_top_p_filtering(output, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)
                # print("next_token.item()",end,next_token.item())
                generate.append(next_token.item())
                generated_out.append(next_token.item())
                prev = next_token.view(1, 1)
                #设置自动停止
                if next_token.item() in end:
                    state="Yes"
                    break
                # print(prev)
        # return generate
        model=None
        inputs=inputs.cpu()
        prev=prev.cpu()
        del inputs
        del prev
        del output
        del past
        del filtered_logits
        del next_token
        del generate
        del model
        torch.cuda.empty_cache()

        # prin
        new =copy(generated_out)
        del generated_out
        gc.collect()
        return new,state


    # 通过命令行参数--fast_pattern，指定模式
    def generate(self,model, context,end, length, temperature=1, top_k=0, top_p=0.0, device='cpu', is_fast_pattern=False):
        # wd=weakref.WeakValueDictionary()
        """
        返回值为
        id列表和释放结束 state
        """
        if is_fast_pattern:
            return self.fast_sample_sequence(model, context,end, length, temperature=temperature, top_k=top_k, top_p=top_p,
                                        device=device)
            # wd['data']=self.fast_sample_sequence(model, context,end, length, temperature=temperature, top_k=top_k, top_p=top_p,
            #                             device=device)
            # return wd['data']
        else:
            return self.sample_sequence(model, context,end, length, temperature=temperature, top_k=top_k, top_p=top_p, device=device)




    # @profile
    def ai(self,text='',args={},key='12312',load_model=None):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("device",device)
        device = "cpu"
        # device =  "cpu"
        model,tokenizer=load_model

        if args.get('length')!=None:
            length=args['length']
        else:
            length=20
        if args.get('nsamples')!=None:
            nsamples=args['nsamples']
        else:
            nsamples=2
        if args.get('start')!=None:
            start=args['start']
        else:
            start=''
        if args.get('end')!=None:
            end=args['end']
        else:
            end="[END]"
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
        parser.add_argument('--length', default=length, type=int, required=False, help='生成长度')
        parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
        parser.add_argument('--nsamples', default=nsamples, type=int, required=False, help='生成几个样本')
        parser.add_argument('--temperature', default=0.8, type=float, required=False, help='生成温度')
        parser.add_argument('--topk', default=10, type=int, required=False, help='最高几选一')
        parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
        # parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
        #                     help='模型参数')
        # parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='词表路径')
        parser.add_argument('--model_config', default='model/mini/config.json', type=str, required=False,
                            help='模型参数')
        parser.add_argument('--tokenizer_path', default='model/mini/vocab.txt', type=str, required=False, help='词表路径')
        parser.add_argument('--model_path', default='model/mini', type=str, required=False, help='模型路径')
        parser.add_argument('--prefix', default=text, type=str, required=False, help='生成文章的开头')
        parser.add_argument('--remove_prefix', default=True, required=False, help='移除头部')
        parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
        parser.add_argument('--segment', action='store_true', help='中文以词为单位')
        parser.add_argument('--fast_pattern',default=True, action='store_true', help='采用更加快的方式生成文本')
        parser.add_argument('--save_samples', action='store_true', help='保存产生的样本')
        parser.add_argument('--save_samples_path', default='.', type=str, required=False, help="保存样本的路径")
        parser.add_argument('--tid', default=key, type=str, required=False, help='保存生成内容')
        parser.add_argument('--end', default=end, type=str, required=False, help="提前结束词")
        parser.add_argument('--start', default=start, type=str, required=False, help="设置开始预测词")
        args = parser.parse_args()
        print('args:\n' + args.__repr__())

        if args.no_wordpiece:
            from tokenizations import tokenization_bert_without_wordpiece as tokenization_bert
        elif args.segment:
            from tokenizations import tokenization_bert_word_level as tokenization_bert
        else:
            from tokenizations import tokenization_bert

        os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
        length = args.length
        batch_size = args.batch_size
        nsamples = args.nsamples
        temperature = args.temperature
        topk = args.topk
        topp = args.topp

        # device = "cuda" if torch.cuda.is_available() else "cpu"

        # tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
        # model = GPT2LMHeadModel.from_pretrained(args.model_path)
        # model.to(device)
        # model.eval()

        if length == -1:
            length = model.config.n_ctx - len(args.prefix)
        # elif length > model.config.n_ctx - len(args.prefix):
        #     # raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)
        #     # raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)
        #     print("输入内容过长自动裁切,方便生成足够数据")
        #     args.prefix=args.prefix[-(model.config.n_ctx-args.length):]
        # if args.fast_pattern and length > model.config.n_ctx - len(args.prefix):
        #     args.prefix=args.prefix[-(model.config.n_ctx-args.length):]


        if args.save_samples:
            if not os.path.exists(args.save_samples_path):
                os.makedirs(args.save_samples_path)
            # samples_file = open(args.save_samples_path + '/samples.txt', 'w', encoding='utf8')
        while True:
            raw_text = args.prefix+""+args.start

            if model.config.n_ctx  < len(args.prefix):
                raw_text=raw_text[-(model.config.n_ctx-3):]
            # raw_text =tkit.Text().clear(args.prefix+'')
            pre_context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
            generated = 0
            all_text=[]
            end=['[END]']

            if args.end!="":
                end.append(args.end)
            data_list = []
            end = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" ".join(end)))
            for _ in range(nsamples // batch_size):
                if args.fast_pattern:
                    new=[]
                    context_tokens=copy(pre_context_tokens)
                    state="No"
                    while len(new)<length and state=="No":
                        if length-len(new)<50:
                            p_length=length-len(new)
                        else:
                            p_length=50
                        if len(context_tokens)>950:
                            context_tokens=context_tokens[-950:]
                        
                        out,state = self.generate(
                            model=model,
                            context=context_tokens,
                            end=end,
                            length=p_length,
                            is_fast_pattern=args.fast_pattern,
                            temperature=temperature, top_k=topk, top_p=topp, device=device
                        )
                        # out_w=weakref.proxy(out)
                        # print('out_w',out_w)
                        # print("out_w num",sys.getrefcount(out_w))
                        context_tokens=context_tokens+out
                        new=new+out
                        del out
                        # for e in end:
                        #     if e in context_tokens:
                        #         # del context_tokens
                        #         print("发现结尾退出")
                        #         break
                    out=new
                else:
                    out,state = self.generate(
                        model=model,
                        context=pre_context_tokens,
                        end=end,
                        length=length,
                        is_fast_pattern=args.fast_pattern,
                        temperature=temperature, top_k=topk, top_p=topp, device=device
                    )                


                for i in range(batch_size):
                    generated += 1
                    text = tokenizer.convert_ids_to_tokens(out)
                    del out
                    text=[args.start]+text
                    # print("text11",text)
                    print("生成长度:",len(text))
                    for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
                        if is_word(item) and is_word(text[i + 1]):
                            text[i] = item + ' '
                    # if args.remove_prefix:
                    #     # print('raw_text',raw_text)
                    #     text=text[-length:]
                        # print('text',text)
                    data={'do':'text','text':''}
                    print("text",text)
                    text_all=''

                    for i, item in enumerate(text):
                        # print(item)
                        if item == '[MASK]':
                            # text[i] = ''
                            pass
                        elif item == '[CLS]' or item == '[SEP]' :
                            # print('缓存')
                           text_all=text_all+ '\n'
                        elif item  == '[END]':
                            # print('缓存')
                           text_all=text_all+ item+' [END] \n\n'
                           break
                        elif item == '[PT]':
                            # print("获取pt")
                            text_all=text_all+ '\n #  [PT] '
                        elif item == '[/PT]':
                            text_all=text_all+ ' [/PT] \n'
                        elif item == '[TT]':
                            # print("获取pt")
                            text_all=text_all+ '\n # [TT]'
                        elif item == '[/TT]':
                            text_all=text_all+ '\n  [/TT] \n'
                        elif item == '[SM]':
                            # print("获取pt")
                            text_all=text_all+ '\n  [SM] '
                        elif item == ' [/SM] \n':
                            text_all=text_all+ ' [/SM] \n'
                        elif item == '[CONTNET]':
                            # print("获取pt")
                            text_all=text_all+ '\n  [CONTNET] '
                        elif item == '[/CONTNET]':
                            text_all=text_all+ ' [/CONTNET] \n'
                        else:
                            text_all=text_all+item
                    text_all = ''.join(text_all).replace('##', '').strip()
                    print("生成长度text_all:",len(text_all))
                    for i, item in enumerate(text):
                        # print(text[i])
                    
                        if item == '[MASK]':
                            text[i] = ''
                        elif item == '[CLS]' or item == '[SEP]':
                            # print('缓存')
                            text[i] = '\n'
                            data['do']="text"
                        elif item  == '[END]':
                            # print('缓存')
                            # text_all=text_all+ '\n\n\n[END]\n\n'
                            break
                           
                        elif item == '[PT]':
                            # print("获取pt")
                            text[i] = ''
                            data['pt']=''
                            data['do']='pt'
                        elif item == '[/PT]':
                            data['do']='none'
                            text[i] = ''
                        elif item == '[TT]':
                            # print("获取pt")
                            text[i] = ''
                            data['tt']=''
                            data['do']='tt'
                        elif item == '[/TT]':
                            data['do']='tt'
                            text[i] = ''
                        else:
                            try:
                                data[data['do']]=data[data['do']]+item
                            except :
                                data[data['do']]=''
                            
        
                    text = ''.join(text_all).replace('##', '').strip()
                    data['all']=text_all
                        # if item == '[title]':
                        #     text[i] = '\n标题: '
                    info = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n"
                    print(info)
                    # print(data)
                    text = ''.join(text).replace('##', '').strip()
                    print(text)
                    data_list.append(data)

                    if text in all_text:
                        # pass
                        print("重复生成")
                    else: 
                        print("添加内容")
                        all_text.append(text)
                        data_pre=get_temp(args.tid)
                        if data_pre['value'].get('text'):
                            data_pre['value']['text'].append(data)
                            pass
                        else:
                            data_pre['value']['text']=[data]
                            pass
                        set_temp(args.tid,data_pre['value'])

            print("=" * 80)
    

            # del out
            
            # for x in locals().keys():
            #     # print("清理函数内存",locals()[x])
            #     del locals()[x]
            # gc.collect()
            # new =copy(all_text)
            del all_text
            gc.collect()
            # return new
            # print(get_temp(args.tid))
            return data_list


            # if generated == nsamples:
            #     # close file when finish writing.
            #     if args.save_samples:
            #         samples_file.close()
            #     break
        # del model,all_text
        # gc.collect()

# q = queue.Queue()
class Writing:
    def __init__(self,args):
        # self.text_list=[]
        self.key= uuid.uuid4()
        self.state=None
        self.args=args
        
        
        pass
    def show(self,arg,se):
        func_name = sys._getframe().f_code.co_name 
        se.acquire()
        # time.sleep(1)
        # print('thread '+str(arg)+" running....")
        ai=Ai()
        model=ai.load_model()
        ai.ai(text=arg,args=self.args,key=self.key,model=model)
        # print("text",text)
        # self.text_list=self.text_list+text
        # q.put((text, func_name)) 
        # print(self.text_list)
        se.release()
        # return self.text_list
    def writing(self,text):
        # 设置允许5个线程同时运行
        self.tt=tiktThreading.TT(1)
        self.tt.load(self.show,text)
        self.tt.start()
        
        # return  tt.t.result     # 获取线程执行结果
        # print(tt)
    def close(self):
        # self.tt.t.close()
        pass
    def get_key(self):
        return self.key
    def get_state(self):
        return self.tt.t.isAlive()
    # def get(self):
    #     result=[]
    #     while not q.empty(): 
    #         print(q.get())
    #         result=result+q.get()
    #         return result

def get_writing(text,args={"length":20}):
    W=Writing(args)
    W.writing(text)
    # print(W.key)
    # print("get_key",W.get_key())
    state=W.get_state()
    while state==True:
        time.sleep(1)
        # print("W.get_state()",W.get_state())
        state=W.get_state()
    W.close()
    return get_temp(W.key)
# def get_pre(key):
#     W=Writing(args)
#     W.writing(text)
#     # print(W.key)
#     # print("get_key",W.get_key())
#     state=W.get_state()
#     while state==True:
#         time.sleep(1)
#         # print("W.get_state()",W.get_state())
#         state=W.get_state()
#     W.close()
#     return get_temp(W.key)
# def ai_title(text='',length=50,nsamples=5,key='默认'):
#     """
#     这里生成kg
#     """
#     print("运行知识提取任务")
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
#     parser.add_argument('--length', default=length, type=int, required=False, help='生成长度')
#     parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
#     parser.add_argument('--nsamples', default=nsamples, type=int, required=False, help='生成几个样本')
#     parser.add_argument('--temperature', default=0.68, type=float, required=False, help='生成温度')
#     parser.add_argument('--topk', default=10, type=int, required=False, help='最高几选一')
#     parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
#     # parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
#     #                     help='模型参数')
#     # parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='词表路径')
#     parser.add_argument('--model_config', default='config/model_config.json', type=str, required=False,
#                         help='模型参数')
#     parser.add_argument('--tokenizer_path', default='cache/vocab.txt', type=str, required=False, help='词表路径')
#     parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
#     parser.add_argument('--prefix', default=text, type=str, required=False, help='生成文章的开头')
#     parser.add_argument('--remove_prefix', default=True, required=False, help='生成文章的开头')
#     parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
#     parser.add_argument('--segment', action='store_true', help='中文以词为单位')
#     parser.add_argument('--fast_pattern',default=True, action='store_true', help='采用更加快的方式生成文本')
#     parser.add_argument('--save_samples', action='store_true', help='保存产生的样本')
#     parser.add_argument('--save_samples_path', default='.', type=str, required=False, help="保存样本的路径")
#     parser.add_argument('--tid', default='0', type=str, required=False, help='保存生成内容')

#     args = parser.parse_args()

#     pre_data= get_var(key)
#     if pre_data['value'].get('do')=="stop":
#         print("已经停止")
#         return []



#     print('args:\n' + args.__repr__())

#     if args.no_wordpiece:
#         from tokenizations import tokenization_bert_without_wordpiece as tokenization_bert
#     elif args.segment:
#         from tokenizations import tokenization_bert_word_level as tokenization_bert
#     else:
#         from tokenizations import tokenization_bert

#     os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
#     length = args.length
#     batch_size = args.batch_size
#     nsamples = args.nsamples
#     temperature = args.temperature
#     topk = args.topk
#     topp = args.topp

#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
#     model = GPT2LMHeadModel.from_pretrained(args.model_path)
#     model.to(device)
#     model.eval()

#     if length == -1:
#         length = model.config.n_ctx - len(args.prefix)
#     elif length > model.config.n_ctx - len(args.prefix):
#         # raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)
#         # raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)
#         print("输入内容过长自动裁切,方便生成足够数据")
#         args.prefix=args.prefix[-(model.config.n_ctx-args.length):]
#     if args.save_samples:
#         if not os.path.exists(args.save_samples_path):
#             os.makedirs(args.save_samples_path)
#         samples_file = open(args.save_samples_path + '/samples.txt', 'w', encoding='utf8')
#     while True:
#         # raw_text = args.prefix
#         raw_text =tkit.Text().clear(args.prefix+'')
#         context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
#         generated = 0
#         all_text=[]
#         for _ in range(nsamples // batch_size):
#             pre_data= get_var(key)
#             if pre_data['value'].get('do')=="stop":
#                 print("已经停止")
#                 return []
#             end=[]
#             out = generate(
#                 model=model,
#                 context=context_tokens,
#                 end=end,
#                 length=length,
#                 is_fast_pattern=args.fast_pattern,
#                 temperature=temperature, top_k=topk, top_p=topp, device=device
#             )
#             for i in range(batch_size):
#                 generated += 1
#                 text = tokenizer.convert_ids_to_tokens(out)
#                 for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
#                     if is_word(item) and is_word(text[i + 1]):
#                         text[i] = item + ' '
#                 kgs=[]
#                 for i, item in enumerate(text):
#                     # print(text[i])
#                     if item == '[MASK]':
#                         text[i] = ''
#                     if item == '[CLS]' or item == '[SEP]':
#                         # print('缓存')
#                         text[i] = '[end] \n'

#                     # [unused5] 标记关键词
#                     # [unused6]  标记标题
#                     # [unused7]  标记前文标题  
#                     # [unused8]  标记正文
#                     # if item == '[unused5]' or item == '[unused6]' or item == '[unused7]' or item == '[unused8]' or item == '[unused9]' ':
#                     #     text[i] = '\n'
#                     # if item == '[TT]':
#                     #     text[i] = ' [keywords] \n'
#                     #     print("关键词")
#                     # if item == '[TT]':
#                     #     text[i] = ' [title] \n'
#                     # if item == '[PT]':
#                     #     text[i] = ' [pretitle] \n'        
#                     # if item == '[unused8]':
#                     #     text[i] = ' [content] \n'      
#                     # if item == '[kgs]':
#                     #     text[i] = ' [content] \n'     
#                     # if item == '[kgs]':
#                     #     text[i] = ' |||'   
#                     # if item == '[kg]':
#                     #     text[i] = ' |||'  
#                     # if item == '[/kg]':
#                     #     text[i] = ' |||'       
#                     # if item == '[kge]':
#                     #     text[i] = ' |||'    
#                     #     break                                            
#                     # if item == '[title]':
#                     #     text[i] = '\n标题: '
#                 info = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n"
#                 print(info)
#                 text = ''.join(text).replace('##', '').strip()

#                 # text = ''.join(text).replace('##', '').strip()
#                 print(text)
#                 kg_start='[pt]'
#                 kg_end='[/pt]'
#                 try:
#                     kg_start_n=text.index(kg_start)

#                 except :
#                     continue
#                 try:

#                     kg_end_n=text.index(kg_end)
                    
#                 except :
#                     try:
#                         kg_end_n=text.index('[end]')
#                     except :
#                         continue
                         
#                 text=text[(kg_start_n+4):kg_end_n]         


#                 # text = text.replace('[/kg]', '||').replace('[kg]', '').replace('[kge]', '').strip()

                
#                 print("提取到的标题：",text)
#                 if args.remove_prefix:
#                     text=text.replace(raw_text,'')
#                 tt=tkitText.Text()
#                 if text in all_text:
#                     pass
#                 else: 
#                     all_text.append(text)
#                     if len(text)>0:
#                         one_data={"_id":tt.md5(text+key),"title":text,'key':key,'time':time.time(),"state":'uncheck'}
#                         #获取之前保存的标题数量  
#                         try:
#                             DB.pre_titles.insert_one(one_data)
#                             pass
#                         except:
#                             pass
#                         else:
#                             pass
#                         pre_data= get_var(key)
#                         num=pre_data['value'].get('num')
#                         if num==None:
#                             pre_data['value']['num']=0
#                         else:
#                             pre_data['value']['num']=pre_data['value']['num']+1
#                         # 保存进程
#                         set_var(key,pre_data['value'])

#         print("=" * 80)
#         del model
#         gc.collect()
#         for x in locals().keys():
#             # print("清理函数内存",locals()[x])
#             del locals()[x]
#         gc.collect()
#         print("获取的所有标题",all_text)
#         return all_text

# def ai_kg(text='',length=20,nsamples=5):
#     """
#     这里生成kg
#     """
#     print("运行知识提取任务")
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='生成设备')
#     parser.add_argument('--length', default=length, type=int, required=False, help='生成长度')
#     parser.add_argument('--batch_size', default=1, type=int, required=False, help='生成的batch size')
#     parser.add_argument('--nsamples', default=nsamples, type=int, required=False, help='生成几个样本')
#     parser.add_argument('--temperature', default=0.7, type=float, required=False, help='生成温度')
#     parser.add_argument('--topk', default=10, type=int, required=False, help='最高几选一')
#     parser.add_argument('--topp', default=0, type=float, required=False, help='最高积累概率')
#     parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False,
#                         help='模型参数')
#     parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False, help='词表路径')
#     parser.add_argument('--model_path', default='model/final_model', type=str, required=False, help='模型路径')
#     parser.add_argument('--prefix', default=text, type=str, required=False, help='生成文章的开头')
#     parser.add_argument('--remove_prefix', default=True, required=False, help='生成文章的开头')
#     parser.add_argument('--no_wordpiece', action='store_true', help='不做word piece切词')
#     parser.add_argument('--segment', action='store_true', help='中文以词为单位')
#     parser.add_argument('--fast_pattern',default=True, action='store_true', help='采用更加快的方式生成文本')
#     parser.add_argument('--save_samples', action='store_true', help='保存产生的样本')
#     parser.add_argument('--save_samples_path', default='.', type=str, required=False, help="保存样本的路径")
#     parser.add_argument('--tid', default='0', type=str, required=False, help='保存生成内容')

#     args = parser.parse_args()
#     print('args:\n' + args.__repr__())

#     if args.no_wordpiece:
#         from tokenizations import tokenization_bert_without_wordpiece as tokenization_bert
#     elif args.segment:
#         from tokenizations import tokenization_bert_word_level as tokenization_bert
#     else:
#         from tokenizations import tokenization_bert

#     os.environ["CUDA_VISIBLE_DEVICES"] = args.device  # 此处设置程序使用哪些显卡
#     length = args.length
#     batch_size = args.batch_size
#     nsamples = args.nsamples
#     temperature = args.temperature
#     topk = args.topk
#     topp = args.topp

#     # device = "cuda" if torch.cuda.is_available() else "cpu"
#     device = "cpu"

#     tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
#     model = GPT2LMHeadModel.from_pretrained(args.model_path)
#     model.to(device)
#     model.eval()

#     if length == -1:
#         length = model.config.n_ctx - len(args.prefix)
#     elif length > model.config.n_ctx - len(args.prefix):
#         # raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)
#         # raise ValueError("Can't get samples longer than window size: %s" % model.config.n_ctx)
#         print("输入内容过长自动裁切,方便生成足够数据")
#         args.prefix=args.prefix[-(model.config.n_ctx-args.length):]
#     if args.save_samples:
#         if not os.path.exists(args.save_samples_path):
#             os.makedirs(args.save_samples_path)
#         samples_file = open(args.save_samples_path + '/samples.txt', 'w', encoding='utf8')
#     while True:
#         # raw_text = args.prefix
#         raw_text =tkit.Text().clear(args.prefix+'')
#         context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(raw_text))
#         generated = 0
#         all_text=[]
#         end=[]
#         for _ in range(nsamples // batch_size):
#             out = generate(
#                 model=model,
#                 context=context_tokens,
#                 end=end,
#                 length=length,
#                 is_fast_pattern=args.fast_pattern,
#                 temperature=temperature, top_k=topk, top_p=topp, device=device
#             )
#             for i in range(batch_size):
#                 generated += 1
#                 text = tokenizer.convert_ids_to_tokens(out)
#                 for i, item in enumerate(text[:-1]):  # 确保英文前后有空格
#                     if is_word(item) and is_word(text[i + 1]):
#                         text[i] = item + ' '
#                 kgs=[]
#                 for i, item in enumerate(text):
#                     # print(text[i])
#                     if item == '[MASK]':
#                         text[i] = ''
#                     if item == '[CLS]' or item == '[SEP]':
#                         # print('缓存')
#                         text[i] = '\n'

#                     # [unused5] 标记关键词
#                     # [unused6]  标记标题
#                     # [unused7]  标记前文标题  
#                     # [unused8]  标记正文
#                     # if item == '[unused5]' or item == '[unused6]' or item == '[unused7]' or item == '[unused8]' or item == '[unused9]' ':
#                     #     text[i] = '\n'
#                     if item == '[TT]':
#                         text[i] = ' [keywords] \n'
#                         print("关键词")
#                     if item == '[TT]':
#                         text[i] = ' [title] \n'
#                     if item == '[PT]':
#                         text[i] = ' [pretitle] \n'        
#                     if item == '[unused8]':
#                         text[i] = ' [content] \n'      
#                     # if item == '[kgs]':
#                     #     text[i] = ' [content] \n'     
#                     # if item == '[kgs]':
#                     #     text[i] = ' |||'   
#                     # if item == '[kg]':
#                     #     text[i] = ' |||'  
#                     # if item == '[/kg]':
#                     #     text[i] = ' |||'       
#                     # if item == '[kge]':
#                     #     text[i] = ' |||'    
#                     #     break                                            
#                     # if item == '[title]':
#                     #     text[i] = '\n标题: '
#                 info = "=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40 + "\n"
#                 print(info)
#                 text = ''.join(text).replace('##', '').strip()

#                 # text = ''.join(text).replace('##', '').strip()
#                 kg_start='[kgs]'
#                 kg_end='[kge]'
#                 kg_start_n=text.index(kg_start)
#                 kg_end_n=text.index(kg_end)
#                 text=text[(kg_start_n+5):kg_end_n]

#                 text = text.replace('[/kg]', '||').replace('[kg]', '').replace('[kge]', '').strip()

                
#                 print(text)
#                 if args.remove_prefix:
 
#                     # remove_prefix_length =len(context_tokens)
#                     # print(remove_prefix_length)
#                     # text=text[remove_prefix_length:]
 
#                     # prefix_clean =tkit.Text().clear(args.prefix)
#                     print('raw_text',raw_text)
#                     text=text.replace(raw_text,'')

#                 if text in all_text:
#                     pass
#                 else: 
#                    all_text.append(text)
#                 if args.save_samples:
#                     samples_file.write(info)
#                     samples_file.write(text)
#                     samples_file.write('\n')
#                     samples_file.write('=' * 90)
#                     samples_file.write('\n' * 2)
#         print("=" * 80)
#         del model
#         gc.collect()
#         for x in locals().keys():
#             # print("清理函数内存",locals()[x])
#             del locals()[x]
#         gc.collect()
#         #保存生成的数据
#         tkit.File().mkdir('tmp')
#         data_path="tmp/run_task"+args.tid+".json"
#         print('保存生成',data_path)
#         tjson=tkit.Json(file_path=data_path)
#         tjson.save([{'prefix':args.prefix,'data':all_text}])

#         return all_text

#         if generated == nsamples:
#             # close file when finish writing.
#             if args.save_samples:
#                 samples_file.close()
#             break
#     # del model,all_text
#     # gc.collect()




def test_gcleak():
    gc.enable()
    gc.set_debug(gc.DEBUG_STATS | gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_INSTANCES)

    print ("begin leak test...")
    # make_circle_ref()
    Ai().ai(text='你好',length=20,nsamples=1)

    print( "\nbegin collect...")
    _unreachable = gc.collect()
    print ("unreachable object num:%d" %(_unreachable))
    print ("garbage object num:%d" %(len(gc.garbage)))


if __name__ == '__main__':
    # main()
    # test_gcleak()
    # ai()
    # gc.set_threshold(200, 10, 5)
    # gc.enable()
    # gc.set_debug(gc.DEBUG_LEAK)
    # text=['柯基犬真是']
    # for i in range(1000):
        # text=Ai().ai(text=text[0],length=20,nsamples=1)
    # Ai().ai()
    # # ai_kg()
    ai = Ai()
    load_model = ai.load_model()

    data=ai.ai(load_model=load_model)
    model,_=load_model
    model.cpu()
    torch.cuda.empty_cache()
    del model
    # print(data)
    # time.sleep(1000)
    
    # W=Writing()
    # W.writing("柯基犬")
    # print(W.key)
    # print("get_key",W.get_key())
    # state=W.get_state()
    # while state==True:
    #     time.sleep(1)
    #     print("W.get_state()",W.get_state())
    #     state=W.get_state()
    # for i in range(100):
    #     args={"end":"[/PT]",'start':'[PT]'}
    #     print(get_writing("柯基犬",args))
        
