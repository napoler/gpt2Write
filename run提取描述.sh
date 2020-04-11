#!/bin/bash

# scrapy crawl itoutiao
# scrapy crawl toutiao
# scrapy crawl itoutiao
# scrapy crawl mtoutiao
while :
do
    # python 转化成为json训练数据.py
    python run提取描述.py
    # scrapy crawl itoutiao
    # scrapy crawl mtoutiao
    # load=`w|head -1|awk -F 'load average: ' '{print $2}'|cut -d. -f1`
    # if [ $load -gt 10 ]
    # then
    #     /usr/local/sbin/mail.py xxx@qq.com "load high" "$load"
    # fi
    sleep 10

done