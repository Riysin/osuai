from data_upload import data_list, thread_start
import asyncio
import time 
  
def test():
    while True:
        print(data_list[0])
        time.sleep(0.1)
        
        
if __name__ == '__main__':
    thread_start()
    time.sleep(1)
    test()
