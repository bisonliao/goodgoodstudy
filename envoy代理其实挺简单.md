# envoy代理其实挺简单

istiod是怎么使用envoy作为sidecar的，可以见另外一篇《istio其实挺简单》

这篇是把envoy当做haproxy nginx这样的独立代理程序来实验各种场景，以解除envoy的神秘。也是因为envoy资料少、配置复杂容易错，所以沉淀一下成功的实验。

### 实验一：代理gRPC请求，且按照gRPC方法路由到不同集群

场景：对grpc进行代理：grpc方法为/echo.Echo的就发往10.11.7.239:8888集群，否则就发往10.11.7.239:50051集群

gRPC协议如下：

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

```yaml
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        address: 0.0.0.0
        port_value: 7890
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          access_log:
            - name: envoy.access_loggers.stdout
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.access_loggers.stream.v3.StdoutAccessLog
          codec_type: auto
          stat_prefix: ingress_http
          route_config:
            name: local_route
            virtual_hosts:
            - name: local_service
              domains: ["*"]
              routes:
              - match:
                  prefix: "/echo.EchoService/Echo" # 如果请求的前缀是/echo.Echo，就路由到echo_cluster
                  grpc: {}
                route:
                  cluster: echo_cluster
              - match:
                  prefix: "/" # 如果请求的前缀是其他的，就路由到other_cluster
                  grpc: {}
                route:
                  cluster: other_cluster
          http_filters:
          - name: envoy.filters.http.grpc_web
          - name: envoy.filters.http.router
  clusters:
  - name: echo_cluster
    connect_timeout: 0.25s
    type: static
    lb_policy: round_robin
    http2_protocol_options: {}
    load_assignment:
      cluster_name: echo_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 10.11.7.239
                port_value: 8888
  - name: other_cluster
    connect_timeout: 0.25s
    type: static
    lb_policy: round_robin
    http2_protocol_options: {}
    load_assignment:
      cluster_name: other_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 10.11.7.239
                port_value: 50051

```

### 实验二：代理gRPC请求，按比例分发请求

80%的请求发送到10.11.7.239:8888集群，20%的请求发往10.11.7.239:50051集群

```yaml
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        address: 0.0.0.0
        port_value: 7890
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          codec_type: auto
          stat_prefix: ingress_http
          route_config:
            name: local_route
            virtual_hosts:
            - name: local_service
              domains: ["*"]
              routes:
              - match:
                  prefix: "/" # 匹配所有的grpc请求
                  grpc: {}
                route:
                  weighted_clusters: # 使用权重集群来分配请求
                    clusters:
                    - name: echo_cluster # 80%的请求转发到echo_cluster
                      weight: 80
                    - name: other_cluster # 20%的请求转发到other_cluster
                      weight: 20
                    total_weight: 100 # 权重总和必须为100
          http_filters:
          - name: envoy.filters.http.grpc_web
          - name: envoy.filters.http.router
          access_log: # 添加access_log配置，将日志打印到stdout
          - name: envoy.access_loggers.stdout
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.access_loggers.stream.v3.StdoutAccessLog
  clusters:
  - name: echo_cluster
    connect_timeout: 0.25s
    type: static
    lb_policy: round_robin
    http2_protocol_options: {} # 添加http2_protocol_options配置，使用http2协议
    load_assignment:
      cluster_name: echo_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 10.11.7.239
                port_value: 8888
  - name: other_cluster
    connect_timeout: 0.25s
    type: static
    lb_policy: round_robin
    http2_protocol_options: {} # 添加http2_protocol_options配置，使用http2协议
    load_assignment:
      cluster_name: other_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 10.11.7.239
                port_value: 50051

```

### 实验三：增加超时和重试

后端服务要保持监听可连接状态，但是不及时回包，制造超时的情形。后台可以抓包发现有每间隔1s重试3次

```yaml
static_resources:
  listeners:
  - name: listener_0
    address:
      socket_address:
        address: 0.0.0.0
        port_value: 7890
    filter_chains:
    - filters:
      - name: envoy.filters.network.http_connection_manager
        typed_config:
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
          codec_type: auto
          stat_prefix: ingress_http
          route_config:
            name: local_route
            virtual_hosts:
            - name: local_service
              domains: ["*"]
              routes:
              - match:
                  prefix: "/" # 匹配所有的grpc请求
                  grpc: {}
                route:
                  weighted_clusters: # 使用权重集群来分配请求
                    clusters:
                    - name: echo_cluster # 80%的请求转发到echo_cluster
                      weight: 80
                    - name: other_cluster # 20%的请求转发到other_cluster
                      weight: 20
                    total_weight: 100 # 权重总和必须为100
                  timeout: 0s # 超时时间
                  retry_policy: # 重试策略
                    retry_on: "5xx" # 重试条件
                    num_retries: 2 # 重试次数
                    per_try_timeout: 1s # 每次重试超时时间
                    retry_host_predicate: # 重试主机谓词
                    - name: envoy.retry_host_predicates.previous_hosts # 谓词名称
                    host_selection_retry_max_attempts: 2 # 主机选择重试最大尝试次数
                    retriable_status_codes: # 可重试状态码
                    - 14 # UNAVAILABLE
          http_filters:
          - name: envoy.filters.http.grpc_web
          - name: envoy.filters.http.router
          access_log: # 添加access_log配置，将日志打印到stdout
          - name: envoy.access_loggers.stdout
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.access_loggers.stream.v3.StdoutAccessLog
  clusters:
  - name: echo_cluster
    connect_timeout: 0.25s
    type: static
    lb_policy: round_robin
    http2_protocol_options: {} # 添加http2_protocol_options配置，使用http2协议
    load_assignment:
      cluster_name: echo_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 10.11.7.239
                port_value: 8888
  - name: other_cluster
    connect_timeout: 0.25s
    type: static
    lb_policy: round_robin
    http2_protocol_options: {} # 添加http2_protocol_options配置，使用http2协议
    load_assignment:
      cluster_name: other_cluster
      endpoints:
      - lb_endpoints:
        - endpoint:
            address:
              socket_address:
                address: 10.11.7.239
                port_value: 50051
```

### 实验 四：熔断

将grpc方法为/echo.Echo的请求发送到后端集群，后端集群有两个endpointer：10.11.7.239:8888和10.11.7.239:50051。希望配置熔断，如果失败率超过10%就熔断1分钟

#### envoy的熔断

envoy的circuit-break更像是限频，我们通常理解的熔断，在envoy这里叫做异常点隔离。

例如下面的配置，像是限频：

```yaml
# envoy.yaml
static_resources: # 静态资源配置
  listeners: # 监听器配置
  - name: listener_0 # 监听器名称
    address: # 地址
      socket_address: # 套接字地址
        protocol: TCP # 协议
        address: 0.0.0.0 # IP地址
        port_value: 7890 # 端口值
    filter_chains: # 过滤器链配置
    - filters: # 过滤器配置
      - name: envoy.filters.network.http_connection_manager # 过滤器名称
        typed_config: # 类型化配置
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager # 类型
          access_log:
            - name: envoy.access_loggers.stdout
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.access_loggers.stream.v3.StdoutAccessLog
          stat_prefix: ingress_http # 统计前缀
          codec_type: AUTO # 编解码器类型
          route_config: # 路由配置
            name: local_route # 路由名称
            virtual_hosts: # 虚拟主机配置
            - name: local_service # 虚拟主机名称
              domains: ["*"] # 域名
              routes: # 路由配置
              - match: # 匹配配置
                  prefix: "/" # 前缀
                  grpc: {} # grpc匹配
                route: # 路由配置
                  cluster: echo_cluster # 集群名称
                  timeout: 0s # 超时时间
                  retry_policy: # 重试策略
                    retry_on: "5xx" # 重试条件
                    num_retries: 2 # 重试次数
                    per_try_timeout: 1s # 每次重试超时时间
                    retry_host_predicate: # 重试主机谓词
                    - name: envoy.retry_host_predicates.previous_hosts # 谓词名称
                    host_selection_retry_max_attempts: 2 # 主机选择重试最大尝试次数
                    retriable_status_codes: # 可重试状态码
                    - 14 # UNAVAILABLE
          http_filters: # HTTP过滤器配置
          - name: envoy.filters.http.router # 过滤器名称
            typed_config: {} # 类型化配置
  clusters: # 集群配置
  - name: echo_cluster # 集群名称
    connect_timeout: 0.25s # 连接超时时间
    type: STATIC # 集群类型
    http2_protocol_options: {}
    lb_policy: ROUND_ROBIN # 负载均衡策略
    circuit_breakers: # 熔断器配置
      thresholds: # 阈值配置
      - priority: DEFAULT # 优先级
        max_connections: 100 # 最大连接数
        max_pending_requests: 100 # 最大等待请求数
        max_requests: 100 # 最大请求数
        max_retries: 3 # 最大重试次数
        retry_budget: # 重试预算
        #budget_percent: 10.0  # 预算百分比
          min_retry_concurrency: 3 # 最小重试并发数
      - priority: HIGH # 优先级
        max_connections: 100 # 最大连接数
        max_pending_requests: 100 # 最大等待请求数
        max_requests: 100 # 最大请求数
        max_retries: 3 # 最大重试次数
        retry_budget: # 重试预算
        #budget_percent: 10.0 # 预算百分比
          min_retry_concurrency: 3 # 最小重试并发数
    load_assignment: # 负载分配配置
      cluster_name: echo_cluster # 集群名称
      endpoints: # 端点配置
      - lb_endpoints: # 负载均衡端点配置
        - endpoint: # 端点
            address: # 地址
              socket_address: # 套接字地址
                address: 10.11.7.239 # IP地址
                port_value: 8888 # 端口值
        - endpoint: # 端点
            address: # 地址
              socket_address: # 套接字地址
                address: 10.11.7.239 # IP地址
                port_value: 50051 # 端口值

```

发现不符合预期，有没有那段熔断相关的配置，envoy都能较好的容错，前端感受不到其中一个endpoint故障。同时可以抓包到envoy非常频繁的在试探故障的endpointer，时间间隔同客户端请求的间隔，报文长度是0,奇怪：

```shell
ubuntu@VM-7-239-ubuntu:~$ ps auxw|grep grpc
ubuntu   3367307  0.0  0.0 457320  8108 pts/0    Sl   13:30   0:00 ./grpcsrv 0.0.0.0:8888
ubuntu   3368219  0.0  0.0 457320  8072 pts/0    Sl   13:34   0:00 ./grpcsrv 0.0.0.0:50051
ubuntu   3369089  0.0  0.0   6300   720 pts/0    S+   13:37   0:00 grep --color=auto grpc
ubuntu@VM-7-239-ubuntu:~$ kill -9 3367307
ubuntu@VM-7-239-ubuntu:~$ sudo tcpdump -i any tcp and \(dst port 8888 \) -nn -v
tcpdump: listening on any, link-type LINUX_SLL (Linux cooked v1), capture size 262144 bytes
13:38:13.288039 IP (tos 0x0, ttl 64, id 39958, offset 0, flags [DF], proto TCP (6), length 60)
    10.11.7.84.40326 > 10.11.7.239.8888: Flags [S], cksum 0x7dce (correct), seq 1303187662, win 64240, options [mss 1424,sackOK,TS val 708600595 ecr 0,nop,wscale 10], length 0
13:38:13.795938 IP (tos 0x0, ttl 64, id 41889, offset 0, flags [DF], proto TCP (6), length 60)
    10.11.7.84.40330 > 10.11.7.239.8888: Flags [S], cksum 0x7d9b (correct), seq 3391853186, win 64240, options [mss 1424,sackOK,TS val 708601103 ecr 0,nop,wscale 10], length 0
13:38:14.319800 IP (tos 0x0, ttl 64, id 43996, offset 0, flags [DF], proto TCP (6), length 60)
    10.11.7.84.40336 > 10.11.7.239.8888: Flags [S], cksum 0x8721 (correct), seq 51924478, win 64240, options [mss 1424,sackOK,TS val 708601627 ecr 0,nop,wscale 10], length 0
13:38:14.835816 IP (tos 0x0, ttl 64, id 57575, offset 0, flags [DF], proto TCP (6), length 60)

```

恢复故障的endpointer后，请求立即打过来，两个endpointer一半一半的请求。

#### envoy的异常点隔离

```yaml
# envoy.yaml
static_resources: # 静态资源配置
  listeners: # 监听器配置
  - name: listener_0 # 监听器名称
    address: # 地址
      socket_address: # 套接字地址
        protocol: TCP # 协议
        address: 0.0.0.0 # IP地址
        port_value: 7890 # 端口值
    filter_chains: # 过滤器链配置
    - filters: # 过滤器配置
      - name: envoy.filters.network.http_connection_manager # 过滤器名称
        typed_config: # 类型化配置
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager # 类型
          access_log:
            - name: envoy.access_loggers.stdout
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.access_loggers.stream.v3.StdoutAccessLog
          stat_prefix: ingress_http # 统计前缀
          codec_type: AUTO # 编解码器类型
          route_config: # 路由配置
            name: local_route # 路由名称
            virtual_hosts: # 虚拟主机配置
            - name: local_service # 虚拟主机名称
              domains: ["*"] # 域名
              routes: # 路由配置
              - match: # 匹配配置
                  prefix: "/" # 前缀
                  grpc: {} # grpc匹配
                route: # 路由配置
                  cluster: echo_cluster # 集群名称
                  timeout: 3s # 超时时间
                  retry_policy: # 重试策略
                    retry_on: "5xx" # 重试条件
                    num_retries: 2 # 重试次数
                    per_try_timeout: 1s # 每次重试超时时间
                    retry_host_predicate: # 重试主机谓词
                    - name: envoy.retry_host_predicates.previous_hosts # 谓词名称
                    host_selection_retry_max_attempts: 2 # 主机选择重试最大尝试次数
                    retriable_status_codes: # 可重试状态码
                    - 14 # UNAVAILABLE
          http_filters: # HTTP过滤器配置
          - name: envoy.filters.http.router # 过滤器名称
            typed_config: {} # 类型化配置
  clusters: # 集群配置
  - name: echo_cluster # 集群名称
    connect_timeout: 0.25s # 连接超时时间
    type: STATIC # 集群类型
    http2_protocol_options: {}
    lb_policy: ROUND_ROBIN # 负载均衡策略
    outlier_detection: # 异常点检测配置
      success_rate_minimum_hosts: 1 # 最小主机数
      success_rate_request_volume: 5 # 请求量
      success_rate_stdev_factor: 1900 # 标准差因子
      interval: 10s # 检测时间间隔
      base_ejection_time: 60s # 基本隔离时间
      max_ejection_percent: 50 # 最大隔离百分比
      enforcing_success_rate: 100 # 强制执行成功率
    load_assignment: # 负载分配配置
      cluster_name: echo_cluster # 集群名称
      endpoints: # 端点配置
      - lb_endpoints: # 负载均衡端点配置
        - endpoint: # 端点
            address: # 地址
              socket_address: # 套接字地址
                address: 10.11.7.239 # IP地址
                port_value: 8888 # 端口值
        - endpoint: # 端点
            address: # 地址
              socket_address: # 套接字地址
                address: 10.11.7.239 # IP地址
                port_value: 50051 # 端口值

```

envoy 的异常点隔离，相当于对单个endpointer的熔断，而实际业务通常是需要对一个service进行熔断。这个不太符合需求。需要在业务层面做熔断。

### 实验五：故障注入

代理的请求，10%的概率丢包，15%的概率延迟0.1s

```yaml
static_resources: # 静态资源配置
  listeners: # 监听器配置
  - name: listener_0 # 监听器名称
    address: # 地址
      socket_address: # 套接字地址
        protocol: TCP # 协议
        address: 0.0.0.0 # IP地址
        port_value: 7890 # 端口值
    filter_chains: # 过滤器链配置
    - filters: # 过滤器配置
      - name: envoy.filters.network.http_connection_manager # 过滤器名称
        typed_config: # 类型化配置
          "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager # 类型
          access_log:
            - name: envoy.access_loggers.stdout
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.access_loggers.stream.v3.StdoutAccessLog
          stat_prefix: ingress_http # 统计前缀
          codec_type: AUTO # 编解码器类型
          route_config: # 路由配置
            name: local_route # 路由名称
            virtual_hosts: # 虚拟主机配置
            - name: local_service # 虚拟主机名称
              domains: ["*"] # 域名
              routes: # 路由配置
              - match: # 匹配配置
                  prefix: "/" # 前缀
                  grpc: {} # grpc匹配
                route: # 路由配置
                  cluster: echo_cluster # 集群名称
          http_filters: # HTTP过滤器配置
          - name: envoy.filters.http.fault
            typed_config:
              "@type": type.googleapis.com/envoy.extensions.filters.http.fault.v3.HTTPFault
              abort:
                percentage:
                  numerator: 10
                  denominator: HUNDRED
                http_status: 503
              delay:
                percentage:
                  numerator: 15
                  denominator: HUNDRED
                fixed_delay: 0.1s
          - name: envoy.filters.http.router # 过滤器名称
            typed_config: {} # 类型化配置
  clusters: # 集群配置
  - name: echo_cluster # 集群名称
    connect_timeout: 0.25s # 连接超时时间
    type: STATIC # 集群类型
    http2_protocol_options: {}
    lb_policy: ROUND_ROBIN # 负载均衡策略
    outlier_detection: # 异常点检测配置
      success_rate_minimum_hosts: 1 # 最小主机数
      success_rate_request_volume: 5 # 请求量
      success_rate_stdev_factor: 1900 # 标准差因子
      interval: 10s # 检测时间间隔
      base_ejection_time: 60s # 基本隔离时间
      max_ejection_percent: 50 # 最大隔离百分比
      enforcing_success_rate: 100 # 强制执行成功率
    load_assignment: # 负载分配配置
      cluster_name: echo_cluster # 集群名称
      endpoints: # 端点配置
      - lb_endpoints: # 负载均衡端点配置
        - endpoint: # 端点
            address: # 地址
              socket_address: # 套接字地址
                address: 10.11.7.239 # IP地址
                port_value: 8888 # 端口值
        - endpoint: # 端点
            address: # 地址
              socket_address: # 套接字地址
                address: 10.11.7.239 # IP地址
                port_value: 50051 # 端口值

```

可以看到，客户端程序的统计到的时延偶尔会大于100ms，部分rpc调用失败：

```shell
Client received: Hello, gRPC! 0
RPC failed with error code: 14, error message: fault filter abort
Client received: Hello, gRPC! 0
RPC failed with error code: 14, error message: fault filter abort
Client received: Hello, gRPC! 0
Client received: Hello, gRPC! 101
Client received: Hello, gRPC! 0
RPC failed with error code: 14, error message: fault filter abort
Client received: Hello, gRPC! 0
Client received: Hello, gRPC! 0
Client received: Hello, gRPC! 104
```

