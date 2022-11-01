

step1: 创建基本工程

```shell
#省去安装maven和java的过程，这个比较容易

mvn archetype:generate  #创建基本工程

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



step2： 引入依赖和插件

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

step3：写一个IDL文件放在 src/main/proto/目录下

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

step4： 编写client

