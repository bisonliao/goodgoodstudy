'''
使用datasketch进行文档相似性检测 http://ekzhu.com/datasketch/lsh.html
数据使用的是wiki中文的1百万条词条（预先分好了词）。内存有限，只能装载几万条。单个文档也比较大，所以索引过程也非常耗时
'''
#coding:utf8
from datasketch import MinHash, MinHashLSH
import pickle

def loadData(filename:str)->(dict, MinHashLSH ):
    docs=dict()
    index = 0
    lsh = MinHashLSH(threshold=0.5)
    with open(filename, "r", encoding="utf8") as f:
        for line in f:
            line = line #type:str
            words = line.split(" ")
            docs[index] = words

            h = MinHash()
            for w in words:
                h.update(w.encode("utf8"))
            lsh.insert(str(index), h)

            index += 1

            print("\rbuilding %d..."%(index), end="")
            if index > 50000:
                break
    print("index finished.")
    with open("./data/SimilarDoc_minhash.pk", "wb") as f:
        pickle.dump((docs, lsh), f)
    return docs, lsh

def test(docs:dict, index:MinHashLSH):
    words = docs[4]
    l = len(words)
    print("len:", l)
    words = words[:1000] # 取一个词条的部分文字作为比对对象
    print(words)
    h = MinHash()
    for w in words:
        h.update(w.encode("utf8"))

    similars = index.query(h)
    for i in similars:
        words = docs[int(i)]
        mh = MinHash()
        for w in words:
            mh.update(w.encode("utf8"))
        d = h.jaccard(mh)
        if d > 0.3:
            print("doc#", i, " similarity:", d, " ", docs[int(i)][:20])
    print("found out similar number:", len(similars))


#docs, lsh = loadData("E:\\DeepLearning\\data\\nlp_data\\corpus\\wiki.zh.corpus.txt")
with open("./data/SimilarDoc_minhash.pk", "rb") as f:
    docs,lsh = pickle.load(f)
test(docs, lsh)
