from multiprocessing import Process,Lock
import os
import time
def addNum(l):
    start = time.time()
    print(start)
    print(os.getpid())
    l.acquire()
    try:
        value = num + 1
        num = value
        print(num)
    finally:
        l.release()
        end = time.time()
        print(end,end-start)
   

if __name__== '__main__':
    num = 0
    lock = Lock()
    print("father process is %s"%(os.getpid()))
    # p = Process(target = printPID,args = ('test',))
    # p.start()
    # p.join()
    # print('mission complete')
    for x in range(5):
        p = Process(target = addNum,args=(lock,))
        p.start()
        p.join()
        # addNum(lock)
    print(num)
