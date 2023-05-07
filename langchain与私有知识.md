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
```

### 亲测llama_index的结果

```shell
把白鹿原第一章内容喂给AI，然后
问：白嘉轩的父亲叫什么？
答：白嘉轩的父亲叫白秉德，他是白鹿原上的名医冷先生的父亲。当冷先生擦拭刀具时，秉德老汉的腿脚手臂松弛下来，口里开始淌出一股乌黑的粘液，令嘉轩感到恶心。秉德老汉在冷先生离开后不久去世，他的死亡是嘉轩头一回经见人的死亡过程

----AI明显混淆了，可能某个地方的“他”没有分清楚是冷先生还是白嘉轩，所以误认为也是冷先生的父亲。

```

