from albert_pytorch import *
import sys
import time
import torch

print(torch.cuda.memory_allocated())
print(torch.cuda.max_memory_allocated())


for i in range(10):
    with torch.no_grad():
        rankclass = classify(model_name_or_path='tkitfiles/rank', num_labels=3, device='cuda')
        p = rankclass.pre("柯基犬喜欢吃")
        print(p)
    rankclass.release()
    # del rankclass
#
#     # print(torch.cuda.memory_cached())
# time.sleep(1)
# model_name_or_path='tkitfiles/rank'
# P = Plus()
# P.args['class_name'] = "AlbertForSequenceClassification"
# P.args['model_name_or_path'] = model_name_or_path
# P.args['finetuning_task'] = 'finetuning_task'
# P.args['num_labels'] = 3
# print(torch.cuda.max_memory_cached())
# model,tokenizer,config_class=P.load_model()
# print(torch.cuda.memory_cached())
# P.release()
# torch.cuda.empty_cache()
# print(torch.cuda.memory_cached())
# time.sleep(100)

# device = torch.device('cuda:0')
# # with torch.no_grad():
# # 定义两个tensor
# dummy_tensor_4 = torch.randn(120, 3, 512, 512).float().to(device)  # 120*3*512*512*4/1000/1000 = 377.48M
# dummy_tensor_2 = torch.randn(80, 3, 512, 512).float().to(device)  # 80*3*512*512*4/1000/1000 = 251.64M
# print(torch.cuda.memory_cached())
# time.sleep(1)
# # 然后释放
# dummy_tensor_4 = dummy_tensor_4.cpu()
# dummy_tensor_2 = dummy_tensor_2.cpu()
# # 这里虽然将上面的显存释放了，但是我们通过Nvidia-smi命令看到显存依然在占用
# print(torch.cuda.memory_cached())
# torch.cuda.empty_cache()
# # 只有执行完上面这句，显存才会在Nvidia-smi中释放
# print(torch.cuda.memory_cached())
time.sleep(100)