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

重试间隔不太符合预期

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
                  timeout: 4s # 设置超时时间为1s
                  retry_policy: # 设置重试策略
                    retry_on: 5xx # 重试条件为5xx错误
                    num_retries: 2 # 重试次数为2
                    retry_back_off: # 设置重试间隔
                      base_interval: 1s # 基础间隔为2s
                      max_interval: 2s # 最大间隔为4s
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

