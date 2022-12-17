import warnings
import requests
import multiprocessing
from colorama import init
import json
init(autoreset=True)

from requests.packages.urllib3.exceptions import InsecureRequestWarning

warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter('ignore', InsecureRequestWarning)

headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
class Worker(multiprocessing.Process):
    def __init__(self, job_queue):
        super().__init__()
        self._job_queue = job_queue
    def run(self):
        while True:
            url = self._job_queue.get()
            if url is None:
                break
            comp = url.strip().split("$$$::")
            link = comp[0]
            name = comp[1]
            try:
                data = requests.get(link)

                # the website: https: // northeastern.sharepoint.com / sites / GenerativeGcode
                gcode_file = f"/Users/alayt/OneDrive/{name}.gcode"
                with open(gcode_file, 'a') as file:
                    file.write(data.text)

            except requests.RequestException as e:
                print('\033[32m' + req + ' - TimeOut!')

if __name__ == '__main__':
    jobs = []
    job_queue = multiprocessing.Queue()
    for i in range(4):
        p = Worker(job_queue)
        jobs.append(p)
        p.start()
    # This is the master code that feeds URLs into queue.
    f = open('data.json')
    data = json.load(f)
    
    for url in data:
        job_queue.put(url)
    # Send None for each worker to check and quit.
    for j in jobs:
        job_queue.put(None)
    for j in jobs:
        j.join()