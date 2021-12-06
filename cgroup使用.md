

### 一、准备

环境：centos

```shell
yum install  libcgroup

vi /etc/cgconfig.conf #增加如下配置，讲cgroup的配置文件加载到指定的虚拟文件系统目录

mount {
    cpuset  = /cgroup/cpuset;
    cpu = /cgroup/cpu;
    cpuacct = /cgroup/cpuacct;
    memory  = /cgroup/memory;
    devices = /cgroup/devices;
    freezer = /cgroup/freezer;
    net_cls = /cgroup/net_cls;
    blkio   = /cgroup/blkio;
}

service cgconfig restart  #重启后就会发现对应的虚拟文件系统目录都创建了，已存在的都是systemd使用到的，不用修改

```



### 二、示例：限制cpu使用份额

```
cgcreate -g cpu:/lesslimit
cgcreate -g cpu:/limit
```

这是会看到在/cgroup/cpu下创建了两个目录：lesslimit和limit，且他们都有很多子目录项

```
cgset -r cpu.shares=1000 lesslimit
cgset -r cpu.shares=100 limit
```

或者直接echo修改文件内容也可以做到：

```
echo 1000 >/cgroup/cpu/lesslimit/cpu.shares
echo 100>/cgroup/cpu/lesslimit/cpu.shares
```

然后写个吃cpu的程序：

```c
//a.c
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main()
{
   float i = 1;
   while (1)
   {
        i = i * 1.2;
   }
}
```

```shell
gcc -o ttx a.c
cp ttx ttx1
cp ttx ttx2
```

然后执行他们

```shell
cgexec -g cpu:limit ./ttx1 &
cgexec -g cpu:lesslimit ./ttx2 &
```

用top可以看到ttx1和ttx2都占用将近100%的cpu，似乎cgroup不起作用:

```shell
  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
24698 root      20   0    7280    372    280 R  98.7  0.0   0:21.85 ttx2
24741 root      20   0    7280    368    280 R  98.0  0.0   0:13.06 ttx1
```

其实是因为我的测试环境有两个核，在资源充分的情况下他们是尽量使用资源。如果多起几个进程，就会发现有效果：

```shell
  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
24698 root      20   0    7280    372    280 R  67.1  0.0   1:06.93 ttx2
24943 root      20   0    7280    372    280 R  56.5  0.0   0:06.18 ttx2
24942 root      20   0    7280    368    280 R  53.8  0.0   0:06.16 ttx2
24741 root      20   0    7280    368    280 R   6.3  0.0   0:44.90 ttx1
24914 root      20   0    7280    368    280 R   6.3  0.0   0:04.48 ttx1
24922 root      20   0    7280    372    280 R   6.3  0.0   0:03.57 ttx1
```

对应还有其他命令：

cgdelete删除资源限制，例如 cgdelete  -g cpu:/lesslimit

cgclear umount掉文件系统

### 三、限制磁盘IO的使用情况

```shell
hongkong1#mount -l
/dev/vda1 on / type ext4 (rw,noatime,data=ordered)

hongkong1#ls -l /dev/vda
brw-rw---- 1 root disk 252, 0 Aug  5 08:33 /dev/vda

hongkong1#echo '252:0 51200' > wahaha/blkio.throttle.read_bps_device #每扇区512B，即每秒读100个扇区

hongkong1#cgexec -g blkio:wahaha time dd if=/root/a.img of=/dev/null  iflag=direct #读一个大文件，跳过缓冲

hongkong1#sar -d 3 100
Linux 3.10.107-1-tlinux2_kvm_guest-0051 (VM_16_7_centos)        12/06/2021      _x86_64_        (2 CPU)

08:42:23 AM       DEV       tps  rd_sec/s  wr_sec/s  avgrq-sz  avgqu-sz     await     svctm     %util
08:42:26 AM  dev252-0    111.33    100.00  10456.00     94.81      0.43      3.89      0.49      5.47
08:42:26 AM   dev11-0      0.00      0.00      0.00      0.00      0.00      0.00      0.00      0.00

08:42:26 AM       DEV       tps  rd_sec/s  wr_sec/s  avgrq-sz  avgqu-sz     await     svctm     %util
08:42:29 AM  dev252-0    114.67    100.00    202.67      2.64      0.05      0.41      0.41      4.67
08:42:29 AM   dev11-0      0.00      0.00      0.00      0.00      0.00      0.00      0.00      0.00

08:42:29 AM       DEV       tps  rd_sec/s  wr_sec/s  avgrq-sz  avgqu-sz     await     svctm     %util
08:42:32 AM  dev252-0    102.33    100.00     58.67      1.55      0.05      0.52      0.52      5.33
08:42:32 AM   dev11-0      0.00      0.00      0.00      0.00      0.00      0.00      0.00      0.00

```

可以从sar看到，每秒钟读100个扇区。

改一下read_bps_device，热更新，立即生效。

### 四、详细的参数说明

详细请见：

```
https://docs.oracle.com/cd/E37670_01/E37355/html/ol_subsystems_cgroups.html
```

