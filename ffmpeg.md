

## 1、ffmpeg的工作流程

完整的流程通常是：解封装、解码、编码、封装，如下图：

```shell
 _______              ______________
|       |            |              |
| input |  demuxer   | encoded data |   decoder
| file  | ---------> | packets      | -----+
|_______|            |______________|      |
                                           v
                                       _________
                                      |         |
                                      | decoded |
                                      | frames  |
                                      |_________|
 ________             ______________       |
|        |           |              |      |
| output | <-------- | encoded data | <----+
| file   |   muxer   | packets      |   encoder
|________|           |______________|
```

输入和输出都可以是多个文件（或者网络）

也可能不编解码，只解封装，再封装，这时候通常对-codec参数指定为copy：

```shell
 _______              ______________            ________
|       |            |              |          |        |
| input |  demuxer   | encoded data |  muxer   | output |
| file  | ---------> | packets      | -------> | file   |
|_______|            |______________|          |________|
```

例如下面的命令，相比重新解码再编码，要快很多：

```shell
ffmpeg -i video.mp4    -t 20  -map 0:0 -c:v copy a.mp4
```



## 2、流的选择

ffmpeg会根据输出文件的后缀自动决定封装格式，也会根据输出文件的封装格式，自动选择流。

还可以通过-map参数手动指定流，输出到目标文件里。例如 -map 1:3，表示选择第二个输入文件的第四条stream

官网上有个很好的例子，直接拷贝：

The following examples illustrate the behavior, quirks and limitations of ffmpeg’s stream selection methods.

They assume the following three input files.

```shell
input file 'A.avi'
      stream 0: video 640x360
      stream 1: audio 2 channels

input file 'B.mp4'
      stream 0: video 1920x1080
      stream 1: audio 2 channels
      stream 2: subtitles (text)
      stream 3: audio 5.1 channels
      stream 4: subtitles (text)

input file 'C.mkv'
      stream 0: video 1280x720
      stream 1: audio 2 channels
      stream 2: subtitles (image)
```



```shell
ffmpeg -i A.avi -i B.mp4 out1.mkv out2.wav -map 1:a -c:a copy out3.mov
```

There are three output files specified, and for the first two, no `-map` options are set, so ffmpeg will select streams for these two files automatically.

out1.mkv is a Matroska container file and accepts video, audio and subtitle streams, so ffmpeg will try to select one of each type.
For video, it will select `stream 0` from B.mp4, which has the highest resolution among all the input video streams.
For audio, it will select `stream 3` from B.mp4, since it has the greatest number of channels.
For subtitles, it will select `stream 2` from B.mp4, which is the first subtitle stream from among A.avi and B.mp4.

out2.wav accepts only audio streams, so only `stream 3` from B.mp4 is selected.

For out3.mov, since a `-map` option is set, no automatic stream selection will occur. The `-map 1:a` option will select all audio streams from the second input B.mp4. No other streams will be included in this output file.

For the first two outputs, all included streams will be transcoded. The encoders chosen will be the default ones registered by each output format, which may not match the codec of the selected input streams.

For the third output, codec option for audio streams has been set to `copy`, so no decoding-filtering-encoding operations will occur, or *can* occur. Packets of selected streams shall be conveyed from the input file and muxed within the output file.

用ffprobe命令可以看文件中的流的情况：

```
> ffprobe a.mp4

Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'a.mp4':
  Metadata:
    major_brand     : isom
    minor_version   : 512
    compatible_brands: isomiso2avc1mp41
    encoder         : Lavf58.29.100
  Duration: 00:00:20.13, start: 0.000000, bitrate: 2089 kb/s
    Stream #0:0(und): Video: h264 (High) (avc1 / 0x31637661), yuv420p, 640x360, 2057 kb/s, 20 fps, 20 tbr, 10240 tbn, 40 tbc (default)
    Metadata:
      handler_name    : Core Media Video
    Stream #0:1(und): Audio: aac (LC) (mp4a / 0x6134706D), 8000 Hz, mono, fltp, 41 kb/s (default)
    Metadata:
      handler_name    : Core Media Audio
```



## 3、常用的一些参数

-i 指定输入文件

-c:v  指定视频编码器

-c:a  指定音频编码器

-t 指定处理的文件的时间长度，单位是秒

-ss 指定媒体截取的开头位置 hh:mm:ss.ms，注意：如果是作用在输入文件上，要写在-i的前面

When used as an input option (before `-i`), seeks in this input file to position. Note that in most formats it is not possible to seek exactly, so `ffmpeg` will seek to the closest seek point before position. When transcoding and -accurate_seek is enabled (the default), this extra segment between the seek point and position will be decoded and discarded. When doing stream copy or when -noaccurate_seek is used, it will be preserved.

When used as an output option (before an output url), decodes but discards input until the timestamps reach position.

position must be a time duration specification, see [(ffmpeg-utils)the Time duration section in the ffmpeg-utils(1) manual](https://ffmpeg.org/ffmpeg-utils.html#time-duration-syntax).

-r 指定视频帧率

-ar 指定音频采样率，例如44.1k

-b:v 指定视频码率

-b:a 指定音频码率

-s 指定视频分辨率，例如1280X720

-ac 指定声道数



看几个例子：

```shell
ffmpeg -ss 00:00:20.000 -i video.mp4 -s 640X360 -c:v h264 -c:a aac -t 20 -r 20 -b:v 2000k -b:a 100k a.mp4
```

只要视频不要音频：

```shell
ffmpeg -ss 00:00:20.000 -i video.mp4 -s 640X360 -c:v h264  -t 20 -r 20 -b:v 2000k -map 0:0 a.mp4
```

## 4、滤波器

简单的filter通过下面两个参数指定

-vf 视频过滤器（视频变换）

-af 音频过滤器（音频变换）

例如通过上下加上灰边（默认是黑边），把视频画面变成1:1

```shell
ffmpeg -i video.mp4  -c:v h264  -t 20  -b:v 2000k -vf 'scale=1280:1280:force_original_aspect_ratio=decrease,pad=1280:1280:(ow-iw)/2:(oh-ih)/2:color=gray' a.mp4
```

复杂的通过-filter_complex指定。

总体上，filter巨复杂，例如降噪、画面叠加等等，可以看：

```shell
https://ffmpeg.org/ffmpeg-filters.html
```



