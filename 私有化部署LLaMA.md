### 私有化部署LLaMA的纪要

##### 1、购买一台海外的云主机，至少8c32G，200GB硬盘，安装好python等，不一定要GPU

##### 2、下载模型的参数。

用wget -r即可

```
https://huggingface.co/decapoda-research
```

##### 3、参考下面的文档，安装webui

注意不需要用webui项目里的download下载，因为前面2已经下载过了

Linux:

1. Follow the [instructions here](https://github.com/oobabooga/text-generation-webui) under "Installation"
2. Download the desired Hugging Face converted model for LLaMA [here](https://huggingface.co/decapoda-research)
3. Copy the entire model folder, for example llama-13b-hf, into text-generation-webui\models
4. Run the following command in your conda environment: python server.py --model llama-13b-hf --load-in-8bit



```
https://www.reddit.com/r/LocalLLaMA/comments/11o6o3f/how_to_install_llama_8bit_and_4bit/
https://github.com/oobabooga/text-generation-webui
```

##### 4、运行webui的server，就可以从浏览器访问了。webui会通过gradio反射出一个域名 

```
python server.py --cpu

Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://88882ef3-a75f-47cb.gradio.live
```





也可以用llma.cpp项目来实现一个命令行工具，使用llama，可以参考：

```
https://juejin.cn/post/7209850311258898490
```

