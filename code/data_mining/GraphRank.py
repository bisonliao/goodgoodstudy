'''
基于二分图的随机游走推荐算法
'''
import csv
import random
import pickle

compute=True

# 加载影评日志，返回
#  1、 二分图
#  2、用户-电影列表 字典，作为测试数据，字典的key是用户id，value是该用户看过的电影id的集合
def load_data():
    G = dict()
    users_test = dict()
    cnt = 0
    with open("E:\\DeepLearning\\data\\movie_rate-20m\\ml-20m\\ratings.csv", "r") as f:
        rate = csv.DictReader(f)
        for rec in rate: # {'rating': '3.5', 'timestamp': '1112486027', 'userId': '1', 'movieId': '2'}
            cnt += 1
            if cnt % 100000 == 0 and cnt > 0:
                print("processed %d records"%(cnt))
            if cnt > 1000000: # 后面user-user相似性的字典会占用很大内存，需要在这里就控制好读取的记录数
                break
            u = int(rec['userId'])
            r = rec['rating']
            m = int(rec['movieId'])
            if float(r) < 4:
                continue
            if random.randint(0,9) > 4:
                key="u_%d"%(u)
                value = "m_%d"%(m)
                if not users_test.__contains__(key):
                    users_test[key] = {value}
                else:
                    users_test[key].add(value)
                continue

            key = "u_%d" % (u)
            value = "m_%d"%(m)
            if not G.__contains__(key) :
                G[key] = {value}
            else:
                G[key].add(value)

            key = "m_%d" % (m)
            value = "u_%d"%(u)
            if not G.__contains__(key):
                G[key] = {value}
            else:
                G[key].add(value)

    return G,users_test

if compute:
    G, users_test = load_data()
    with open("./data/G.data", "wb") as f:
        pickle.dump(G, f)
    with open("./data/users_test.data", "wb") as f:
        pickle.dump(users_test, f)
else:
    with open("./data/G.data", "rb") as f:
        G = pickle.load(f)
    with open("./data/users_test.data", "rb") as f:
        users_test = pickle.load(f)

print("size:%d,%d"%(len(G),len(users_test)) )

# 随机游走，给用户person推荐物品
def PersonRank(person,G, topk = 100):
    alpha = 0.2
    num = 1000
    if not G.__contains__(person):
        return set()
    key = person
    result = dict()
    for e in range(num):
        while True:
            # person -> items
            if not G.__contains__(key):
                break
            itemlist = list(G[key])
            index = random.randint(0, len(itemlist)-1)
            item = itemlist[index]

            if random.random() < alpha: # stop ?
                if result.__contains__(item):
                    result[item] = result[item]+1
                else:
                    result[item] = 1
                break

            # item -> persons
            if not G.__contains__(item):
                break
            personlist = list(G[item])
            index = random.randint(0, len(personlist)-1)
            key = personlist[index]
    sortList = sorted(result.items(), key=lambda x:x[1], reverse=True) #按value（到达概率）降序排列
    #print(">>", sortList)
    result = set()
    for i in range(topk):
        if i >= len(sortList):
            break
        (item, cnt) = sortList[i]
        result.add(item)
    return result

cnt = 0
for (u, movielist) in users_test.items():
    rec = PersonRank(u,G)
    if len(rec) == 0 or len(movielist) == 0:
        continue
    matchsz = len(rec.intersection(movielist))
    print("%s\t%.2f\t%.2f"%(u, matchsz/len(rec), matchsz/len(movielist)))
    cnt += 1
    if cnt > 1000:
        break


