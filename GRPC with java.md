

### step1: 创建基本工程

```shell
#省去安装maven和java的过程，这个比较容易

mvn archetype:generate -DgroupId=org.example -DartifactId=Channel -DarchetypeArtifactId=maven-archetype-quickstart -DinteractiveMode=false  #创建基本工程，注意修改groupId和artifactID

calc/
├── pom.xml
├── src
│   ├── main
│   │   ├── java
│   │   │   └── io
│   │   │       └── bison
│   │   │           └── calc
│   │   │               └── App.java

mvn compile #编译

java -cp ./target/calc-1.0-SNAPSHOT.jar  io.bison.calc.App #执行
```



### step2： 引入依赖和插件

```xml
<?xml version="1.0" encoding="UTF-8"?>

<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
  <modelVersion>4.0.0</modelVersion>

  <groupId>bison</groupId>
  <artifactId>calc</artifactId>
  <version>1.0-SNAPSHOT</version>

  <name>calc</name>
  <!-- FIXME change it to the project's website -->
  <url>http://www.example.com</url>

  <properties>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
    <maven.compiler.source>1.7</maven.compiler.source>
    <maven.compiler.target>1.7</maven.compiler.target>
  </properties>

  <dependencies>
    <dependency>
      <groupId>junit</groupId>
      <artifactId>junit</artifactId>
      <version>4.11</version>
      <scope>test</scope>
    </dependency>

<!-- for protoc begin  bison手动添加的-->
    <dependency>
        <groupId>io.grpc</groupId>
        <artifactId>grpc-bom</artifactId>
        <version>1.50.2</version>
        <type>pom</type>
        <scope>import</scope>
      </dependency>
  <dependency>
    <groupId>io.grpc</groupId>
    <artifactId>grpc-netty-shaded</artifactId>
    <version>1.50.2</version>
    <scope>runtime</scope>
  </dependency>
  <dependency>
    <groupId>io.grpc</groupId>
    <artifactId>grpc-protobuf</artifactId>
    <version>1.50.2</version>
  </dependency>
  <dependency>
    <groupId>io.grpc</groupId>
    <artifactId>grpc-stub</artifactId>
    <version>1.50.2</version>
  </dependency>
  <dependency> <!-- necessary for Java 9+ -->
    <groupId>org.apache.tomcat</groupId>
    <artifactId>annotations-api</artifactId>
    <version>6.0.53</version>
    <scope>provided</scope>
  </dependency>
  <!-- for protoc end -->
  </dependencies>

  <build>
    <extensions>
    <!-- for protoc begin bison手动添加的 -->
      <extension>
        <groupId>kr.motd.maven</groupId>
        <artifactId>os-maven-plugin</artifactId>
        <version>1.6.2</version>
      </extension>
    <!-- for protoc end -->
    </extensions>
    
      <plugins>
        <!-- clean lifecycle, see https://maven.apache.org/ref/current/maven-core/lifecycles.html#clean_Lifecycle -->
        <plugin>
          <artifactId>maven-clean-plugin</artifactId>
          <version>3.1.0</version>
        </plugin>
        <!-- default lifecycle, jar packaging: see https://maven.apache.org/ref/current/maven-core/default-bindings.html#Plugin_bindings_for_jar_packaging -->
        <plugin>
          <artifactId>maven-resources-plugin</artifactId>
          <version>3.0.2</version>
        </plugin>
        <plugin>
          <artifactId>maven-compiler-plugin</artifactId>
          <version>3.8.0</version>
        </plugin>
        <plugin>
          <artifactId>maven-surefire-plugin</artifactId>
          <version>2.22.1</version>
        </plugin>
        <plugin>
          <artifactId>maven-jar-plugin</artifactId>
          <version>3.0.2</version>
        </plugin>
        <plugin>
          <artifactId>maven-install-plugin</artifactId>
          <version>2.5.2</version>
        </plugin>
        <plugin>
          <artifactId>maven-deploy-plugin</artifactId>
          <version>2.8.2</version>
        </plugin>
        <!-- site lifecycle, see https://maven.apache.org/ref/current/maven-core/lifecycles.html#site_Lifecycle -->
        <plugin>
          <artifactId>maven-site-plugin</artifactId>
          <version>3.7.1</version>
        </plugin>
        <plugin>
          <artifactId>maven-project-info-reports-plugin</artifactId>
          <version>3.0.0</version>
        </plugin>

    <!-- for protoc begin bison手动添加的 -->
        <plugin>
          <groupId>org.xolstice.maven.plugins</groupId>
          <artifactId>protobuf-maven-plugin</artifactId>
          <version>0.6.1</version>
          <configuration>
              <protocArtifact>com.google.protobuf:protoc:3.6.1:exe:${os.detected.classifier}</protocArtifact>
              <pluginId>grpc-java</pluginId>
              <pluginArtifact>io.grpc:protoc-gen-grpc-java:1.50.2:exe:${os.detected.classifier}</pluginArtifact>
          </configuration>
          <executions>
            <execution>
               <goals>
                 <goal>compile</goal>
                <goal>compile-custom</goal>
              </goals>
            </execution>
          </executions>
        </plugin>

      <!-- for protoc end -->
      </plugins>

  </build>
</project>

```

### step3：写一个IDL文件放在 src/main/proto/目录下

```protobuf
syntax = "proto3";

option java_multiple_files = true;
option java_package = "io.bison.calc";
option java_outer_classname = "CalcProto";
option objc_class_prefix = "";

package calc;

service Calculator {
    rpc Add(AddRequest) returns (AddResponse){}
    rpc Sub(SubRequest) returns (SubResponse){}
}

message AddRequest {
    int32 a = 1;
    int32 b = 2;
}
message AddResponse {
    int32 result = 1;
}

message SubRequest {
    int32 a = 1;
    int32 b = 2;    
}
message SubResponse {
    int32 result = 1;
}
```

编译

```shell
mvn compile

# 目录下产生文件
target/generated-sources/
├── annotations
└── protobuf
    ├── grpc-java
    │   └── io
    │       └── bison
    │           └── calc
    │               └── CalculatorGrpc.java
    └── java
        └── io
            └── bison
                └── calc
                    ├── AddRequest.java
                    ├── AddRequestOrBuilder.java
                    ├── AddResponse.java
                    ├── AddResponseOrBuilder.java
                    ├── CalcProto.java
                    ├── SubRequest.java
                    ├── SubRequestOrBuilder.java
                    ├── SubResponse.java
                    └── SubResponseOrBuilder.java

```

### step4： 编写client

为了分发更简单，可以把依赖库都打包到jar包里，在pom.xml的plugins里添加一个插件即可：

```xml
     <plugin>
            <groupId>org.apache.maven.plugins</groupId>
            <artifactId>maven-assembly-plugin</artifactId>
            <version>2.4.1</version>
            <configuration>
                <!-- get all project dependencies -->
                <descriptorRefs>
                    <descriptorRef>jar-with-dependencies</descriptorRef>
                </descriptorRefs>
                <!-- MainClass in mainfest make a executable jar 
                这样你就可以直接 java -jar xxx.jar来执行了
					-->
                <archive>
                    <manifest>
                        <mainClass>io.bison.calc.App</mainClass>
                    </manifest>
                </archive>
            </configuration>
            <executions>
                <!-- 配置执行器 -->
                <execution>
                    <id>make-assembly</id>
                    <!-- 绑定到package命令的生命周期上 -->
                    <phase>package</phase>
                    <goals>
                        <!-- 只运行一次 -->
                        <goal>single</goal>
                    </goals>
                </execution>
            </executions>
      </plugin>

# mvn package就会把依赖库一起打包进来
```

要特别注意，maven生成的pom.xml文件，在build和plugins之间，有一个pluginManagement，一定要删除它，否则上面的依赖打包的指示不会生效。他妈的搞得老子折腾2个小时！

```xml
 <build>
    <pluginManagement><!-- 一定要删除这个屌毛 -->
      <plugins>

```



客户端代码很简单：

```java
package io.bison.calc;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import java.util.logging.Logger;

/**
 * Hello world!
 *
 */
public class App 
{
    private static final Logger logger = Logger.getLogger(App.class.getName());

    public static int add(CalculatorGrpc.CalculatorBlockingStub stub, int a, int b)
    {
        AddRequest req = AddRequest.newBuilder().setA(a).setB(b).build();
        try
        {
            AddResponse resp = stub.add(req);
            return resp.getResult();
        }
        catch (StatusRuntimeException e)
        {
            logger.info(""+e.getMessage());
            e.printStackTrace();
            return -1;
        }
        
    }

    public static int sub(CalculatorGrpc.CalculatorBlockingStub stub, int a, int b)
    {
        SubRequest req = SubRequest.newBuilder().setA(a).setB(b).build();
        try
        {
            SubResponse resp = stub.sub(req);
            return resp.getResult();
        }
        catch (StatusRuntimeException e)
        {
            logger.info(""+e.getMessage());
            e.printStackTrace();
            return -1;
        }
        
    }
    public static void main( String[] args )
    {
        System.out.println( "Hello World!" );

        if (args.length < 1)
        {
            System.out.println("App [server ip:port]");
            return;
        }
        String target = args[0];
        ManagedChannel channel = ManagedChannelBuilder.forTarget(target).usePlaintext().build();

        CalculatorGrpc.CalculatorBlockingStub stub = CalculatorGrpc.newBlockingStub(channel);

        System.out.println("1 + 2 = " + add(stub, 1, 2));
        System.out.println("56-32 = " + sub(stub, 56, 32));

        channel.shutdown();

    }
}

```

### step5：编写server

```java
package io.bison.calc;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import java.util.logging.Logger;

public class CalcServer
{
    private final static Logger logger = Logger.getLogger(Server.class.getName());

    public io.grpc.Server srv = ServerBuilder.forPort(50012).addService(new CalcImpl()).build();;

    public static void main(String[] args) throws  Exception{
        CalcServer calc = new CalcServer();
        calc.srv.start();
        calc.srv.awaitTermination();

    }
    class CalcImpl extends CalculatorGrpc.CalculatorImplBase {

        public void add(io.bison.calc.AddRequest request,
                    io.grpc.stub.StreamObserver<io.bison.calc.AddResponse> responseObserver) {
            int result = request.getA() + request.getB();
            AddResponse resp = AddResponse.newBuilder().setResult(result).build();
            responseObserver.onNext(resp);
            responseObserver.onCompleted();
        }

  
        public void sub(io.bison.calc.SubRequest request,
            io.grpc.stub.StreamObserver<io.bison.calc.SubResponse> responseObserver) {
            int result = request.getA() - request.getB();
            SubResponse resp = SubResponse.newBuilder().setResult(result).build();
            responseObserver.onNext(resp);
            responseObserver.onCompleted();
        }

    }
}
```



运行客户端，会遇到一个问题：

```
Caused by: io.grpc.netty.shaded.io.netty.channel.AbstractChannel$AnnotatedConnectException: connect(..) failed: Invalid argument: /127.0.0.1:50012
```

网上是和mvn把依赖库都打包，其中关于DNS的依赖有问题导致的。

```
https://github.com/grpc/grpc-java/issues/9367
```

改用这种方式能够正常运行：

```
mvn exec:java -Dexec.mainClass="io.bison.calc.App" -Dexec.args="127.0.0.1:50012"
```

或者把依赖包grpc-netty-shaded的版本降低到1.30.0，也能正常work。



### 最后：参考资料：

```
https://grpc.io/docs/languages/java/
```

### grpc with c++也搞一下

```shell
sudo apt install g++
sudo apt install protobuf-compiler-grpc
sudo apt install protobuf-compiler
sudo apt install libprotobuf-dev

sudo apt-get install libgrpc++-dev
sudo apt-get install libgrpc-dev
```

echo.proto

```protobuf
syntax = "proto3";

package echo;

service EchoService {
  rpc Echo(EchoRequest) returns (EchoResponse) {}
}

message EchoRequest {
  string message = 1;
}

message EchoResponse {
  string message = 1;
}
```

grpccli.cpp

```c++
#include <grpcpp/grpcpp.h>
#include "echo.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using echo::EchoRequest;
using echo::EchoResponse;
using echo::EchoService;

class EchoClient {
public:
    EchoClient(std::shared_ptr<Channel> channel)
        : stub_(EchoService::NewStub(channel)) {}

    std::string Echo(const std::string& message) {
        EchoRequest request;
        request.set_message(message);

        EchoResponse response;
        ClientContext context;

        Status status = stub_->Echo(&context, request, &response);

        if (status.ok()) {
            return response.message();
        } else {
            std::cout << "RPC failed with error code: " << status.error_code()
                      << ", error message: " << status.error_message()
                      << std::endl;
            return "";
        }
    }

private:
    std::unique_ptr<EchoService::Stub> stub_;
};

int main() {
    std::string server_address("10.11.7.239:50051");
    EchoClient client(grpc::CreateChannel(server_address,
                                          grpc::InsecureChannelCredentials()));

    int i;
    for (i = 0; i < 100000; ++i)
    {
    std::string message = "Hello, gRPC!";
    std::string response = client.Echo(message);
    std::cout << "Client received: " << response << std::endl;
    }

    return 0;
}

```

grpcsrv.cpp

```c++
#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "echo.grpc.pb.h"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::Status;
using echo::EchoRequest;
using echo::EchoResponse;
using echo::EchoService;

// 实现EchoService
class EchoServiceImpl final : public EchoService::Service {
    Status Echo(ServerContext* context, const EchoRequest* request,
                EchoResponse* response) override {
        // 直接将客户端发送的字符串作为响应返回
        response->set_message(request->message());
        return Status::OK;
    }
};

void RunServer() {
    std::string server_address("0.0.0.0:50051");
    EchoServiceImpl service;

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    std::unique_ptr<Server> server(builder.BuildAndStart());
    std::cout << "Server listening on " << server_address << std::endl;

    server->Wait();
}

int main() {
    RunServer();
    return 0;
}

```

build.sh

```shell
protoc -I. --cpp_out=. --grpc_out=. --plugin=protoc-gen-grpc=`which grpc_cpp_plugin` echo.proto
g++ -std=c++11 grpccli.cpp echo.grpc.pb.cc echo.pb.cc -o grpccli `pkg-config --cflags --libs grpc++ grpc` -lprotobuf -pthread
g++ -std=c++11 grpcsrv.cpp echo.grpc.pb.cc echo.pb.cc -o grpcsrv `pkg-config --cflags --libs grpc++ grpc` -lprotobuf -pthread
```

测试10万次执行的耗时：

```shell
ubuntu@VM-7-84-ubuntu:~$ time ./grpccli  >/tmp/bison

real    0m12.426s
user    0m1.484s
sys     0m1.055s
```

大多数时间是在等待网络， rpc本身占用cpu很少，约2.5s， **那如果不考虑网络同步等待，可以认为单纯rpc的性能可以达到4万次每秒**。