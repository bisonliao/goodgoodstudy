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

##### 5、遇到奇怪的问题

1. 当使用30B模型的时候，内存至少要198G
2. 当加载模型参数的时候，如果提示Tokenizer class LLaMATokenizer does not exist or is not currently imported. 要修改 tokenizer_config.json文件里的"tokenizer_class"值为 "LlamaTokenizer"

##### 6、运行效果

```c
//////////////////////////////我让他写一段ffmpeg转码的代码，算是有模有样：（还没有写完，我不等了）
I write a c function to convert ogg to mp3, this function use ffmpeg library::

\begin{code}
int convert_ogg(char *input_file, char *output_file) {
    AVFormatContext* formatCtx = NULL;
    int ret = 0;
    AVCodec* codec = NULL;
    AVStream* stream = NULL;
    FILE* input = fopen(input_file,"rb");
    if (!input) {
        printf("Can't open file '%s'\n", input_file);
        return -1;
    }
    /* find the input stream */
    avformat_open_input(&formatCtx, input_file, NULL, NULL);
    if (avformat_find_stream_info(formatCtx,NULL)) {
        printf("Could not find stream information\n");
        goto end;
    }
    /* select the video stream */
    codec = avcodec_find_decoder(AVCODEC_ID_OGG);
    if (!codec) {
        printf("Codec not found\n");
        goto end;
    }
    stream = formatCtx->streams[0];
    /* allocate and init decoding context */
    ret = avcodec_open2(stream->codec, codec, NULL);
    if (ret < 0) {
        printf("Could not open codec\n");
        goto end;
    }
    /* uncomment next line for debugging */
    //fflush(stdout);
    /* decode one frame of audio data */
    while (av_read_frame(formatCtx, &packet)>=0) {
        if (packet.stream_index == stream->index && packet.flags & AV_PKT_FLAG_KEY) {
            break;
        }
        avcodec_decode_audio4(stream->codec, pcm, &got_frame, &packet);
        if (got_frame) {
            printf("%d %f\n", got_frame, packet.pts);
            /* save
            
/////////////////////我让llama给做一下单元测试，算是有点内容，但明显不怎么聪明：

I write some unit test code below  to test this c function:

int swapIfGT(int *a, int *b)
{
   if (*a > *b) 
  {
      int tmp = *a;
     *a = *b;
     *b = tmp;
     return 1;
  }
  else { return 0;}
}}

\begin{code}
#include <stdio.h>
#include "swap_ifgt.h"

void swap_ifgt_test()
{
    int a=5, b=3;
    printf("before swap: %d,%d\n", a, b);
    swapIfGT(&a, &b);
    printf("after swap: %d,%d\n", a, b);
}
\end{code}

But the output is:

\begin{blockquote}

before swap: 5,3

  after swap: 5,3
\end{blockquote}
```



也可以用llma.cpp项目来实现一个命令行工具，使用llama，可以参考：

```
https://juejin.cn/post/7209850311258898490
```

