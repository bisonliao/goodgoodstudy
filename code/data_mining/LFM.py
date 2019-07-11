import csv
import random
import pickle
import mxnet.ndarray as nd
import mxnet as mx
import mxnet.autograd as autograd

compute=False
# 限于内存大小，通过限制最大的用户Id和电影Id，取部分数据
maxUserID = 20000
maxMovieID = 3000
factorNum = 100 #隐藏因子的个数
context = mx.gpu(0)
epochs = 20000
lr = 0.00001 #由于backup（）没有带batchsz，所以lr很小，而且要根据训练数据的大小适当调整

# 从开源的电影评价数据集中构建评价矩阵用于训练和测试
def load_data():
    num = 0
    with open("E:\\DeepLearning\\data\\movie_rate-20m\\ml-20m\\ratings.csv", "r") as f:
        rate = csv.DictReader(f)
        train_data = nd.zeros(shape=[maxUserID, maxMovieID], ctx = context)
        test_data = nd.zeros(shape=[maxUserID, maxMovieID], ctx=context)
        flags = nd.zeros(shape=[maxUserID, maxMovieID], ctx=context)
        for rec in rate: # {'rating': '3.5', 'timestamp': '1112486027', 'userId': '1', 'movieId': '2'}
            num += 1
            if num % 100000 == 0:
                print("processed %d record"%(num))
            if num > 1000000:
                break
            r = float(rec['rating'])
            u = int(rec['userId'])
            m = int(rec['movieId'])
            if u >= maxUserID or m >= maxMovieID:
                continue
            if random.randint(0, 10) > 7 : #部分用于测试集
                test_data[u,m] = r
            else:
                train_data[u, m] =  r
                flags[u,m] = 1
        return train_data, flags, test_data

if compute:
    train_data, flags, test_data = load_data()
    with open("./data/ratings_train.data", "wb") as f:
        pickle.dump(train_data, f)
    with open("./data/ratings_test.data", "wb") as f:
        pickle.dump(test_data, f)
    with open("./data/flags.data", "wb") as f:
        pickle.dump(flags, f)
else:
    with open("./data/ratings_train.data", "rb") as f:
        train_data = pickle.load(f)
    with open("./data/ratings_test.data", "rb") as f:
        test_data = pickle.load(f)
    with open("./data/flags.data", "rb") as f:
        flags = pickle.load(f)



rate_num = nd.sum(flags).asscalar()
print("rate numbers:", rate_num)

def my_loss(y, label, flags):
    nameda = 0.0001
    return nd.sum((y - label)*(y-label)*flags)   #+nd.sum(nameda * U*U) + nd.sum(nameda*V*V) #后面两项是L2正则项防止过拟合

# 根据评价矩阵，训练出U和V满足 U.V = train_data
def train(train_data, test_data, flags, epochs, start_ep = 0):
    U = nd.random.uniform(0, 1, shape=(maxUserID, factorNum), ctx=context)
    V = nd.random.uniform(0, 1, shape=(factorNum, maxMovieID), ctx=context)
    if start_ep > 0:
        with open("./data/U.data." + str(start_ep), "rb") as f:
            U = pickle.load(f)
        with open("./data/V.data." + str(start_ep), "rb") as f:
            V = pickle.load(f)
    for e in range(start_ep+1, epochs):
        U.attach_grad()
        V.attach_grad()
        with autograd.record():
            Y = nd.dot(U,V)
            L = my_loss(Y, train_data, flags)
        L.backward()
        if e % 100 == 0 and e > 0:
            avgLoss = L.asscalar()/rate_num
            print("ep:%d, loss:%.4f"%(e,avgLoss))
            if avgLoss < 0.01:
                break
        U = U - lr * U.grad
        V = V - lr * V.grad
        if e % 1000 == 0 and e > 0:
            acc = test2(test_data, U, V)
            print("test acc:", acc)
            with open("./data/U.data."+str(e), "wb") as f:
                pickle.dump(U, f)
            with open("./data/V.data."+str(e), "wb") as f:
                pickle.dump(V, f)
    return U,V
def test2(test_data, U, V):
    total = 0
    right_num = 0
    for u in range(maxUserID):
        for m in range(maxMovieID):
            label = test_data[u, m].asscalar()
            if label > 0:
                total += 1
                if total > 1000:
                    return right_num / total
                y = nd.dot(U[u, :].reshape((1, factorNum)), V[:, m].reshape((factorNum, 1)))
                y = y[0, 0].asscalar()
                right = 1 if y / label > 0.9 and y / label < 1.1 else 0  # 只要预测的评分不要差的太多，我都算你对
                #print("%.2f %.2f %d" % (label, y, right))
                if right > 0:
                    right_num += 1
    return right_num / total

def test(test_data, ep):
    with open("./data/U.data." + str(ep), "rb") as f:
        U = pickle.load(f)
    with open("./data/V.data." + str(ep), "rb") as f:
        V = pickle.load(f)
    return test2(test_data, U, V)





U,V = train(train_data, test_data,flags,epochs, 0)
#print(test(test_data, 19000))



