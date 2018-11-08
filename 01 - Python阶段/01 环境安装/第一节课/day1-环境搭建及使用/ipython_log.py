# IPython log file

[关于编译器与解释器的区别](http://blog.csdn.net/touzani/article/details/1625760)
print ("hello, world")
a = 1
a
b = [1, 2, 3]
get_ipython().magic('lsmagic')
get_ipython().magic('whos')
get_ipython().magic('reset -f')
get_ipython().magic('whos')
get_ipython().magic('pwd')
get_ipython().magic('mkdir demo_test')
get_ipython().magic('cd demo_test/')
get_ipython().run_cell_magic('writefile', 'hello_world.py', 'print ("hello world")')
get_ipython().magic('ls')
get_ipython().magic('run hello_world.py')
import os
os.remove('hello_world.py')
get_ipython().magic('ls')
get_ipython().magic('cd ..')
get_ipython().magic('rmdir demo_test')
get_ipython().magic('hist')
get_ipython().magic('pinfo sum')
# 导入numpy和matplotlib两个包
get_ipython().magic('pylab')
# 查看其中sort函数的帮助
get_ipython().magic('pinfo2 sort')
# 导入numpy和matplotlib两个包
get_ipython().magic('pylab')
# 查看其中sort函数的帮助
get_ipython().magic('pinfo2 sort')
a = 12
a
_ + 13
get_ipython().system('ping baidu.com')
$logstart
get_ipython().magic('logstart')
