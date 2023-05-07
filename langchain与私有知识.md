### 问题提出

我有一组内部文档，包含大量的内部知识，如何利用chatGPT这样的大语言模型进行知识检索。

用embedding和KNN是一个方案，这里还有另外一个方案：

### 解决方案

把私有文档分割为chunk，分别embedding后存储到向量数据库。

输入一个查询，embedding后与文档的向量数据库做相似性比较，将相似性较高的chunk提取用作prompt优化，查询大语言模型。

引用网友的一个图：

![](img/llm/langchain.png)

### 参考资料

```shell
https://github.com/imClumsyPanda/langchain-ChatGLM
https://zhuanlan.zhihu.com/p/613155165
https://github.com/jerryjliu/llama_index
https://github.com/facebookresearch/faiss
```

### 亲测结果

#### llama_index

```shell
把白鹿原第一章内容喂给AI，然后
问：白嘉轩的父亲叫什么？
答：白嘉轩的父亲叫白秉德，他是白鹿原上的名医冷先生的父亲。当冷先生擦拭刀具时，秉德老汉的腿脚手臂松弛下来，口里开始淌出一股乌黑的粘液，令嘉轩感到恶心。秉德老汉在冷先生离开后不久去世，他的死亡是嘉轩头一回经见人的死亡过程

----AI明显混淆了，可能某个地方的“他”没有分清楚是冷先生还是白嘉轩，所以误认为也是冷先生的父亲。

```

#### chatGLM：

```shell
docker run --gpus all  -d --name chatglm -v /root/.cache:/root/.cache  -v /root/private:/private chatglm-cuda:latest  sh -c 'sleep 10000000'

root@23dec2396b03:/chatGLM# python3 ./cli_demo.py
Input your local knowledge file path 请输入本地知识文件路径：/private
加载文件: 100%|
文件加载完毕，正在生成向量库
Input your question 请输入问题：白嘉轩的职业是什么
白嘉轩的职业是白鹿原上一位富有的地主和娶过七房女人的多事男子。他通过购买鹿子霖家的土地获得了成功，并且娶到了多任妻子。他的人生充满了娶妻和生育，但是他也因为自己的贪心和冲动而面临了危险。
```

