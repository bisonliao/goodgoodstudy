#### 1、命令行工具

##### 1.1 对称加密

```shell
openssl enc -e -aes-128-cbc -K '11223344556677889900112233445566' -iv '00000000000000000000000000000000' -in .profile -out xxx.dat 
openssl enc -d -aes-128-cbc -K '11223344556677889900112233445566' -iv '00000000000000000000000000000000' -in xxx.dat
openssl enc --help
```

#### 1.2 消息摘要

```shell
openssl dgst --help
openssl dgst -md5 .profile
openssl dgst -sha256 .profile
```

#### 1.3 生成DH公私钥

```shell
openssl dhparam -outform PEM -out xxx.dat  -2
openssl dhparam -inform PEM -in xxx.dat  -C

openssl dhparam --help
```

#### 1.4 生成ECDH公私钥

```shell
openssl ecparam -help
openssl ecparam -list_curves
openssl ecparam -genkey    -name secp521r1
```

#### 1.5 生成DSA 公私钥

```shell
openssl dsaparam  -genkey -outform PEM -out xxx.dat 1024
openssl dsa -in xxx.dat  -pubout
```

