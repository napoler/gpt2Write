import  gc
import  sys
from copy import  copy
# gc.set_debug(gc.DEBUG_STATS|gc.DEBUG_LEAK)
class A:
    def __del__(self):
        pass
class B:
    def __del__(self):
        pass
 
a=A()
b=B()
# print hex(id(a))
# print hex(id(a.__dict__))
print(id(a))
print(id(b))
a.b=b
c=copy(a)
d=a
print(sys.getrefcount(d))
print(sys.getrefcount(c))
print(sys.getrefcount(a))
b.a=a
del a
del b
n=0
for i in range(100000):
    i=i+1
print(sys.getrefcount(i))
print (gc.collect(),gc.garbage)
# print (gc.garbage)
# ————————————————
# 版权声明：本文为CSDN博主「yueguanghaidao」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/yueguanghaidao/article/details/11274737