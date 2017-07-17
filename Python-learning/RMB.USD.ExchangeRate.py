#需要安装lxml库，可通过PyCharm自动安装
#输出人民币兑美元汇率实时信息
import re
from lxml import etree
import requests

url = 'http://www.boc.cn/sourcedb/whpj/index.html'
html = requests.get(url).content.decode('utf8')  # 获取网页源码

loc = html.index('<td>美元</td>')
content = html[loc:loc + 300]# 截取内容
res = re.findall('<td>(.*?)</td>', content)#正则获取

with open('人民币美元汇率.txt', 'w+') as f:
    f.write(res[0] + '\n')
    f.write('现汇买入价：' + res[1] + '\n')
    f.write('现钞买入价：' + res[2] + '\n')
    f.write('现汇卖出价：' + res[3] + '\n')
    f.write('现钞卖出价：' + res[4] + '\n')
    f.write('中行折算价：' + res[5] + '\n')
    f.write('发布时间：' + res[6] + ' ' +res[7])