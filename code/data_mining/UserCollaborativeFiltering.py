'''
使用协同过滤器算法，根据用户的影评数据，给用户推荐可能喜欢的电影。
'''
import csv
import random
import pickle

compute=True

# 加载影评日志，返回
# 1、用户-电影列表 字典  ，字典的key是用户id，value是该用户看过的电影id的集合
# 2、电影-用户列表 字典  ， 字典的key是电影id，value是看过该电影的用户id的集合
# 3、用户-电影列表 字典，作为测试数据，字典的key是用户id，value是该用户看过的电影id的集合
def load_data():
    users = dict()
    users_test = dict()
    movies = dict()
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
            if random.randint(0,9) == 7:
                if not users_test.__contains__(u):
                    users_test[u] = {m}
                else:
                    users_test[u].add(m)
                continue

            if not users.__contains__(u) :
                users[u] = {m}
            else:
                users[u].add(m)
            if not movies.__contains__(m) :
                movies[m] = {u}
            else:
                movies[m].add(u)
    return users,movies,users_test

if compute:
    users, movies, users_test = load_data()
    with open("./data/users.data", "wb") as f:
        pickle.dump(users, f)
    with open("./data/movies.data", "wb") as f:
        pickle.dump(movies, f)
    with open("./data/users_test.data", "wb") as f:
        pickle.dump(users_test, f)
else:
    with open("./data/users.data", "rb") as f:
        users = pickle.load(f)
    with open("./data/movies.data", "rb") as f:
        movies = pickle.load(f)
    with open("./data/users_test.data", "rb") as f:
        users_test = pickle.load(f)

print("size:%d,%d"%(len(users),len(movies)) )

# 根据  电影-用户列表 ，返回 用户-用户的相似性
# 字典的key是用户id，value一个字典，里面保存了与该用户的相似 用户和相似度
def find_similar_user(movies,users):
    similar_users = dict()
    number = 0
    for (k,userlist) in movies.items():
        for u1 in userlist:
            for u2 in userlist:
                if u1 == u2:
                    continue
                number += 1
                #if number % 1000000 == 0:
                    #print("processed %d u-u with same movies"%(number))
                if not similar_users.__contains__(u1):
                    similar_users[u1] = {u2:1}
                else:
                    cnt = 0
                    if similar_users[u1].__contains__(u2):
                        cnt = similar_users[u1][u2]
                    similar_users[u1][u2] = cnt+1
    print("u-u ralation size:%d"%(number))
    number = 0
    for (u1, sim_userlist) in similar_users.items():
        set1 = users[u1]
        for (u2, cnt) in sim_userlist.items():
            number += 1
            #if number % 1000000 == 0:
                #print("processed %d u-u similarity"%(number))
            set2 = users[u2]
            unionsz = len(set1.union(set2))
            similar_users[u1][u2] = cnt / unionsz
    return similar_users

if compute:
    similar_users = find_similar_user(movies,users)
    with open("./data/similar_users.data", "wb") as f:
        pickle.dump(similar_users,f)
else:
    with open("./data/similar_users.data", "rb") as f:
        similar_users = pickle.load(f)

print("similar_users size:", len(similar_users))

# 根据前面的数据，生成推荐的电影列表
def recommend(u, similar_users, users, k=5):
    sim_userlist = sorted(similar_users[u].items(), key=lambda x:x[1], reverse=True) #按value（相似度）降序排列
    print(sim_userlist)
    cnt = 0
    items=set()
    for (u2,v) in sim_userlist:
        cnt += 1
        if cnt > k:
            break
        items = items.union(users[u2])
    return items

rec1551 = recommend(1551, similar_users, users, 1)
label = users_test[1551]
print("recommend:", rec1551)
print("test:", label)
print("match:", label.intersection(rec1551))










