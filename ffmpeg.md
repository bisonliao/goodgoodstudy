

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



## 5、两个有意思的应用

### 5.1 把一个mp4文件模拟成不断推流的直播HLS

```shell
ffmpeg -re -stream_loop -1 -i 0729.mp4 -c:v h264_nvenc -c:a aac -f hls -hls_time 6 -hls_list_size 5 -hls_flags delete_segments+append_list output2/output.m3u8
```

这条命令使用了ffmpeg工具，它的各个参数的意义如下：

- `-re`：以原始帧率读取输入。
- `-stream_loop -1`：无限循环输入流。
- `-i 0729.mp4`：指定输入文件为0729.mp4。
- `-c:v h264_nvenc`：指定视频编码器为h264_nvenc。
- `-c:a aac`：指定音频编码器为aac。
- `-f hls`：指定输出格式为HLS。
- `-hls_time 6`：指定每个HLS分段的持续时间为6秒。
- `-hls_list_size 5`：指定HLS播放列表中最多包含5个分段。
- `-hls_flags delete_segments+append_list`：指定HLS标志为delete_segments和append_list，表示删除旧的分段并将新的分段追加到播放列表中。
- `output2/output.m3u8`：指定输出文件为output2/output.m3u8。

有时候，我们需要把生成的HLS文件及时上传到COS等云存储里，作为CDN的源，那就需要这样一段python代码：

```python
import os
import sys
import ffmpeg
from google.cloud import storage
from google.api_core.exceptions import NotFound
import time
import m3u8
import re

def check_file_modifiedtime(indexfile:str)->int:
    # Path to the file/directory
    path = indexfile
    ti_m = os.path.getmtime(path)
    return ti_m

def get_ts_file_from_indexfile(indexfile:str)->list:
    playlist = m3u8.load(indexfile)  # this could also be an absolute filename
    return playlist.segments

def upload_to_gcs(bucket_name, source_folder, destination_folder):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    mtime = 0
    index = "output.m3u8"
    indexfile = source_folder + "/" + index
    pattern = re.compile(r'#EXT.*\n')
    uploaded = set()
    while (True):
        mtime2 = check_file_modifiedtime(indexfile)
        if mtime2 <= mtime:
            time.sleep(1)
            continue
        mtime = mtime2
        time.sleep(1)
        segments = get_ts_file_from_indexfile(indexfile)
        fileInM3U8 = set()
        blob = bucket.blob(os.path.join(destination_folder, index))
        blob.upload_from_filename(indexfile)
        for seg in segments:
            seg = str(seg)
            seg = re.sub(pattern, "", seg)
            fileInM3U8.add(seg)
            if uploaded.__contains__(seg):
                continue
            print(seg)
            segfile = source_folder + "/" + seg
            blob = bucket.blob(os.path.join(destination_folder, seg))
            blob.upload_from_filename(segfile)
            uploaded.add(seg)
        #delete file that in object storage but not in m3u8 file
        diff = uploaded.difference(fileInM3U8)
        print("diffsize:", diff.__len__())
        for seg in diff:
            uploaded.remove(seg)
            blob = bucket.blob(os.path.join(destination_folder, seg))
            blob.delete()

if __name__ == "__main__":
    input_file = "./0729.mp4"
    output_dir = "./output2"
    stream_id = "0729"
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'zegocdn.json'
    upload_to_gcs('cdn-source-by-bison', output_dir, stream_id)

```

### 5.2 生成带当前时间的视频流用于测试端到端的时延

opencv基于本地的一个black.jpg作为背景，不断生成带有当前时间的图片，通过管道输送给ffmpeg生成HLS直播文件到一个目录

```python
import cv2
import subprocess
import time

# 设置帧率和总帧数
fps = 30
frame_count = 6000

# 设置字体和颜色
font = cv2.FONT_HERSHEY_SIMPLEX
color = (255, 255, 255)

# 启动ffmpeg进程
command = ['ffmpeg',
           '-y',
           '-f', 'rawvideo',
           '-pix_fmt', 'bgr24',
           '-video_size', '1280x720',
           '-vcodec', 'rawvideo',
           '-s', '640x480',
           '-framerate', str(fps),
           '-i', '-',
           '-c:v', 'h264',
           '-b:v', '1M',
           '-pix_fmt', 'yuv420p',
           '-hls_time', '3',
           '-hls_list_size', '3',
           '-f', 'hls',
           'output2/output.m3u8']
ffmpeg = subprocess.Popen(command, stdin=subprocess.PIPE)

# 生成视频帧
for i in range(frame_count):
    # 创建一个黑色背景的图像
    img = cv2.imread('black.jpg')
    # 获取当前时间并绘制到图像上
    current_time = time.strftime("%H:%M:%S", time.localtime())
    cv2.putText(img, current_time, (100, 100), font, 2, color, 2)
    # 将图像数据写入ffmpeg的标准输入
    ffmpeg.stdin.write(img.tobytes())
    # 等待1/fps秒
    time.sleep(1.0 / fps)

# 关闭ffmpeg的标准输入
ffmpeg.stdin.close()
# 等待ffmpeg进程结束
ffmpeg.wait()
```

需要安装的库有：

```shell
apt-get install ffmpeg
apt-get install libx264-dev
apt-get install x26
apt install python3-pi
apt-get install python3-opencv
apt-get install ffmpeg

pip3 install ffmpeg-python
pip3 instal scikit-build
pip3 install opencv-python
pip3 install m3u
```

