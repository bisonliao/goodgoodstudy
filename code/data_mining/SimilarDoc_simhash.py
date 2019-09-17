'''
使用simhash模块(https://leons.im/posts/a-python-implementation-of-simhash-algorithm/)对文档进行相似性比较
数据使用的是wiki中文的1百万条词条（预先分好了词）。内存有限，只能装载几万条。单个文档也比较大，所以索引过程也非常耗时

simhash page: http://www.wwwconference.org/www2007/papers/paper215.pdf
'''
from simhash import Simhash
from simhash import SimhashIndex
import pickle

#coding:utf8

def loadData(filename:str)->(dict, SimhashIndex ):
    docs=dict()
    index = 0
    # about parameter f and k, the paper said:
    # "...In a collection of f-bit fingerprints, quickly
    # find all fingerprints that differ from a given fingerprint in at
    # most k bit positions, where k is a small integer..."
    # k越大，召回率越高，准确率越低
    search = SimhashIndex([],k= 6)
    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            line = line #type:str
            words = line.split(" ")
            docs[index] = words
            search.add(str(index), Simhash(words, f=64))

            print("\rbuilding %d..." % (index), end="")
            index += 1
            if index > 50000:
                break

    # SimhashIndex支持用add方法增加文档索引
    # 也可以在构造函数里直接传入文档列表：
    #search = SimhashIndex([(str(k), Simhash(v, f=64)) for k,v in docs.items()], k=9)

    print("index finished.")
    with open("./data/SimilarDoc_simhash.pk", "wb") as f:
        pickle.dump((docs, search), f)
    return docs,search
def test(docs:dict, index:SimhashIndex):
    words = docs[4]
    l = len(words)
    print("len:", l)
    words = words[:800] # 取一个词条的部分文字作为比对对象
    print(words)
    h = Simhash(words)
    similars = index.get_near_dups(h)
    for i in similars:
        d = h.distance(Simhash(docs[int(i)]))
        if d < 7:
            print("doc#", i, " distance:", d, " ", docs[int(i)][:20])
    print("found out similar number:", len(similars))


docs, index = loadData("E:\\DeepLearning\\data\\nlp_data\\corpus\\wiki.zh.corpus.txt")

with open("./data/SimilarDoc_simhash.pk", "rb") as f:
    docs,index = pickle.load(f)

test(docs, index)