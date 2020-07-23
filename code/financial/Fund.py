# -*- coding: utf-8 -*-

# get fund net value dataset from http://fund.eastmoney.com
# check if it is profitable to invest some fund each month

import requests
import datetime
import time
import execjs
from lxml import etree
import os.path
import pickle
import math



class Fund:

    def __init__(self, fundCode:str):
        self.fundCode = fundCode
        self.fundvalue = {} #type:dict
        filename = "./data/%s_v2.fund"%(fundCode)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.fundCode, self.fundname, self.fundtype, self.fundmanager, self.fundteam, self.fundvalue = pickle.load(f)
        else:
            self.__download()
            self.__get_info()
            towrite = (self.fundCode, self.fundname, self.fundtype, self.fundmanager, self.fundteam, self.fundvalue)
            with open(filename, 'wb') as f:
                pickle.dump(towrite, f)


    def __download(self):
        # http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code=161616&page=40&sdate=2002-01-01&edate=2020-07-17&per=49
        now = datetime.datetime.now()
        today = now.strftime("%Y-%m-%d")
        #print(today)
        self.fundvalue = {}
        for page in range(1,1000):
            url = "http://fund.eastmoney.com/f10/F10DataApi.aspx?type=lsjz&code=%s&page=%d&sdate=2000-01-01&edate=%s&per=20"%(self.fundCode, page, today)
            response = requests.get(url)
            response = response.text #type:str
            #print(response)
            if response.__contains__("暂无数据"):
                break
            object = execjs.eval(response.replace("var ", "").replace(";","").replace("table ", "table id='FundNetValue' "))
            curpage = object["curpage"]
            records = object["records"]
            pages = object["pages"]
            doc = etree.HTML("<html><body>"+object["content"]+"</body></html>")
            recnum = 20
            if curpage == pages: #last page
                recnum = records % 20

            for row in range(1, recnum+1):
                #print(doc.xpath("//table[@id=\"FundNetValue\"]/tbody/tr[%d]//td/text()"%(row)))
                dt = doc.xpath("//table[@id=\"FundNetValue\"]/tbody/tr[%d]//td[1]/text()"%(row))
                netval = doc.xpath("//table[@id=\"FundNetValue\"]/tbody/tr[%d]//td[2]/text()"%(row))
                sumval = doc.xpath("//table[@id=\"FundNetValue\"]/tbody/tr[%d]//td[3]/text()" % (row))

                if len(netval) == 1:
                    netval = netval[0]
                else:
                    print("no netval!", response)
                    continue
                #some fund has no sum value
                if len(sumval) == 1:
                    sumval = sumval[0]
                else:
                    sumval = netval

                self.fundvalue[dt[0]] = {"netval":netval, "sumval":sumval}

            if pages <= page:
                break
            #time.sleep(1) # avoid to frequently request the website

    def __get_info(self):

        url = "http://fund.eastmoney.com/%s.html"%(self.fundCode)
        response = requests.get(url)
        s = response.content
        doc = etree.HTML(s)
        self.fundname = doc.xpath("//span[@class='funCur-FundName']/text()")[0]
        self.fundtype = doc.xpath("//td[contains(text(),'基金类型')]/a/text()")[0]
        self.fundmanager = doc.xpath("//td[contains(text(),'基金经理：')]/a/text()")[0]
        self.fundteam = doc.xpath("//span[contains(text(),'管 理 人')]/following-sibling::a/text()")[0]




    def invest(self, smart=False, stopprofit=False, maxcnt=0):
        dt = [*self.fundvalue.keys()]
        dt.sort()
        year_mon = ""
        share = 0.0
        cost = 0.0
        reward=0.0
        lastday = ""

        netvalsum = 0.0
        netvalcnt = 0
        investcnt = 0

        for day in dt:
            if day[0:4] == "2020": # 2020 first half year, there is a boom in stock/fund market, get rid of this influence.
                break
            if maxcnt>0 and investcnt > maxcnt:
                break
            lastday = day
            currentval = float(self.fundvalue[day]["netval"])
            netvalcnt += 1
            netvalsum += currentval
            if day[0:7] != year_mon:
                #print(day, self.fundvalue[day])
                investcnt += 1
                year_mon = day[0:7]
                if smart and netvalcnt > 30 and (currentval / (netvalsum/netvalcnt)) < 0.9:
                    # if netval is lower than avg, invest double
                    cost += 2000
                    share += 2000.0 / currentval

                else:
                    cost += 1000.0
                    share += 1000.0 / currentval
            if stopprofit:
                if cost>0 and (currentval*share / cost) >= 1.3:
                    reward += currentval*share
                    share = 0 # restart

        if cost < 1000:
            return 0, 0, 0

        reward += float(self.fundvalue[lastday]["netval"])*share

        years = investcnt / 12.0
        times =  reward / cost
        rate = math.pow(10, math.log10(times) / years)

        return times, years , rate


def __main__():


    '''code = ["164402", "160643", "502003", "502010", "501048", "257060", "005542",
            "001891", "161616", "004075", "519736", "002803" , "110003", "007784",
            "160620", "005911", "710302", "501010", "270023", "004973", "003745",
            "006751",  "001717","004851", "001915", "320007", "002939", "519674",
            "005911", "002560", "000297", "005461", "001045", "710301", "710302",
            "161726", "007300", "007301", "501009", "501010", "001984", "002891",
            "001691", "006308", "270023", "004860", "004749", "004973", "003468",
            "002302", "004503", "002719", "007378", "007377", "002552", "161838",
            "009411", "009857", "009858", "009770", "161223", "519677", "000478",
            "110050", "000727", "160716", "070023", "159916", "512600", "110022"]'''
    code=["159932", "159915", "512330", "510290", "000942", "510500", "510050", "510880", "510510", "000248"]



    total = 0.0
    cnt = 0
    high = []
    medium = []
    low = []
    goodcnt = 0
    for c in code:
        f = Fund(c)

        #(t, y, r) = f.automatic_invest()
        (t, y, r) = f.invest(False, True)
        print("%s, %.02f, %.02f, %.03f"%(c,t,y,r) )
        total += r*y
        cnt += y
        if r >= 1.06:
            goodcnt += 1
        if r >= 1.15:
            high.append(f.fundname)
        elif r >= 1.06:
            medium.append(f.fundname)
        else:
            low.append(f.fundname)
    print("invest by random:", total/ cnt)
    print("有中高回报：%d/%d"%(goodcnt, len(code)))

    print("高回报:", high)
    print("中回报:", medium)
    print("低回报:", low)


__main__()