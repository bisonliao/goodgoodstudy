## HTTP/2

### 简介

GRPC是基于HTTP/2的，用tcpdump抓包的时候，发现HTTP/2的协议和HTTP/1.x 完全不一样，有这么一些特点：

1. 二进制，而不是文本的。对于开发者可读性较差，但对于计算机来说解析效率更高、传输效率更高
2. 多路复用，一个tcp连接可以有上百个stream同时存在和复用
3. 流控，有基于stream和该tcp连接全局的流控，类似tcp协议的流控，滑动窗口
4. stream间有不同的优先级，高优先级的frame获得更多的带宽资源和传输机会
5. 头部压缩，每个stream不用重复传输http头部
6. server push，不同于WebSocket的全双工，HTTP/2的server push不能单独由server发起
7. 默认基于TLS。TCP 三次握手后，TLS用两个RTT建立TLS会话，在TLS的两个RTT中同时完成HTTP/2的协商。

非常像QUIC



### 参考资料

《Http2 In Action》

《Learning HTTP2 a practical guide for beginners》

### 开发库

搜了一下，比较有名的是nghttp2这个库，基于c语言，文档也非常详细。这个库类似QUIC协议下的msquic，不管IO，只负责协议解析，通过异步回调的方式与上层应用交互。

The most notable point in nghttp2 library architecture is it does not perform any I/O. nghttp2 only performs HTTP/2 protocol stuff based on input byte strings. It will call callback functions set by applications while processing input. The output of nghttp2 is just byte string. An application is responsible to send these output to the remote peer. The callback functions may be called while producing output.

```
https://nghttp2.org/documentation/programmers-guide.html
```

我尝试把官网的一个client的示例代码修改为我自己的代码，但是失败了，从网络上收到报文，回填给nghttp2这个库，不能触发收到应用数据的回调。但使用官网的示例代码是能够正常工作的。（待解决，2022年10月6日）

网上搜了一下，java 8默认提供了对http/2的支持，相比之下，库的使用对开发者也更友好。

这是nghttp2的官网示例代码，基于libevent实现网络IO，可以正常工作：

```c
/*
 * nghttp2 - HTTP/2 C Library
 *
 * Copyright (c) 2013 Tatsuhiro Tsujikawa
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifdef __sgi
#  include <string.h>
#  define errx(exitcode, format, args...)                                      \
    {                                                                          \
      warnx(format, ##args);                                                   \
      exit(exitcode);                                                          \
    }
#  define warnx(format, args...) fprintf(stderr, format "\n", ##args)
char *strndup(const char *s, size_t size);
#endif

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <sys/types.h>
#ifdef HAVE_UNISTD_H
#  include <unistd.h>
#endif /* HAVE_UNISTD_H */
#ifdef HAVE_SYS_SOCKET_H
#  include <sys/socket.h>
#endif /* HAVE_SYS_SOCKET_H */
#ifdef HAVE_NETINET_IN_H
#  include <netinet/in.h>
#endif /* HAVE_NETINET_IN_H */
#include <netinet/tcp.h>
#ifndef __sgi
#  include <err.h>
#endif
#include <signal.h>
#include <string.h>

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/conf.h>

#include <event.h>
#include <event2/event.h>
#include <event2/bufferevent_ssl.h>
#include <event2/dns.h>

#include <nghttp2/nghttp2.h>

#include "url_parser.h"

#define ARRLEN(x) (sizeof(x) / sizeof(x[0]))

typedef struct {
  /* The NULL-terminated URI string to retrieve. */
  const char *uri;
  /* Parsed result of the |uri| */
  struct http_parser_url *u;
  /* The authority portion of the |uri|, not NULL-terminated */
  char *authority;
  /* The path portion of the |uri|, including query, not
     NULL-terminated */
  char *path;
  /* The length of the |authority| */
  size_t authoritylen;
  /* The length of the |path| */
  size_t pathlen;
  /* The stream ID of this stream */
  int32_t stream_id;
} http2_stream_data;

typedef struct {
  nghttp2_session *session;
  struct evdns_base *dnsbase;
  struct bufferevent *bev;
  http2_stream_data *stream_data;
} http2_session_data;

static http2_stream_data *create_http2_stream_data(const char *uri,
                                                   struct http_parser_url *u) {
  /* MAX 5 digits (max 65535) + 1 ':' + 1 NULL (because of snprintf) */
  size_t extra = 7;
  http2_stream_data *stream_data = malloc(sizeof(http2_stream_data));

  stream_data->uri = uri;
  stream_data->u = u;
  stream_data->stream_id = -1;

  stream_data->authoritylen = u->field_data[UF_HOST].len;
  stream_data->authority = malloc(stream_data->authoritylen + extra);
  memcpy(stream_data->authority, &uri[u->field_data[UF_HOST].off],
         u->field_data[UF_HOST].len);
  if (u->field_set & (1 << UF_PORT)) {
    stream_data->authoritylen +=
        (size_t)snprintf(stream_data->authority + u->field_data[UF_HOST].len,
                         extra, ":%u", u->port);
  }

  /* If we don't have path in URI, we use "/" as path. */
  stream_data->pathlen = 1;
  if (u->field_set & (1 << UF_PATH)) {
    stream_data->pathlen = u->field_data[UF_PATH].len;
  }
  if (u->field_set & (1 << UF_QUERY)) {
    /* +1 for '?' character */
    stream_data->pathlen += (size_t)(u->field_data[UF_QUERY].len + 1);
  }

  stream_data->path = malloc(stream_data->pathlen);
  if (u->field_set & (1 << UF_PATH)) {
    memcpy(stream_data->path, &uri[u->field_data[UF_PATH].off],
           u->field_data[UF_PATH].len);
  } else {
    stream_data->path[0] = '/';
  }
  if (u->field_set & (1 << UF_QUERY)) {
    stream_data->path[stream_data->pathlen - u->field_data[UF_QUERY].len - 1] =
        '?';
    memcpy(stream_data->path + stream_data->pathlen -
               u->field_data[UF_QUERY].len,
           &uri[u->field_data[UF_QUERY].off], u->field_data[UF_QUERY].len);
  }

  return stream_data;
}

static void delete_http2_stream_data(http2_stream_data *stream_data) {
  free(stream_data->path);
  free(stream_data->authority);
  free(stream_data);
}

/* Initializes |session_data| */
static http2_session_data *
create_http2_session_data(struct event_base *evbase) {
  http2_session_data *session_data = malloc(sizeof(http2_session_data));

  memset(session_data, 0, sizeof(http2_session_data));
  session_data->dnsbase = evdns_base_new(evbase, 1);
  return session_data;
}

static void delete_http2_session_data(http2_session_data *session_data) {
  SSL *ssl = bufferevent_openssl_get_ssl(session_data->bev);

  if (ssl) {
    SSL_shutdown(ssl);
  }
  bufferevent_free(session_data->bev);
  session_data->bev = NULL;
  evdns_base_free(session_data->dnsbase, 1);
  session_data->dnsbase = NULL;
  nghttp2_session_del(session_data->session);
  session_data->session = NULL;
  if (session_data->stream_data) {
    delete_http2_stream_data(session_data->stream_data);
    session_data->stream_data = NULL;
  }
  free(session_data);
}

static void print_header(FILE *f, const uint8_t *name, size_t namelen,
                         const uint8_t *value, size_t valuelen) {
  fwrite(name, 1, namelen, f);
  fprintf(f, ": ");
  fwrite(value, 1, valuelen, f);
  fprintf(f, "\n");
}

/* Print HTTP headers to |f|. Please note that this function does not
   take into account that header name and value are sequence of
   octets, therefore they may contain non-printable characters. */
static void print_headers(FILE *f, nghttp2_nv *nva, size_t nvlen) {
  size_t i;
  for (i = 0; i < nvlen; ++i) {
    print_header(f, nva[i].name, nva[i].namelen, nva[i].value, nva[i].valuelen);
  }
  fprintf(f, "\n");
}

/* nghttp2_send_callback. Here we transmit the |data|, |length| bytes,
   to the network. Because we are using libevent bufferevent, we just
   write those bytes into bufferevent buffer. */
static ssize_t send_callback(nghttp2_session *session, const uint8_t *data,
                             size_t length, int flags, void *user_data) {
  http2_session_data *session_data = (http2_session_data *)user_data;
  struct bufferevent *bev = session_data->bev;
  (void)session;
  (void)flags;

  bufferevent_write(bev, data, length);
  return (ssize_t)length;
}

/* nghttp2_on_header_callback: Called when nghttp2 library emits
   single header name/value pair. */
static int on_header_callback(nghttp2_session *session,
                              const nghttp2_frame *frame, const uint8_t *name,
                              size_t namelen, const uint8_t *value,
                              size_t valuelen, uint8_t flags, void *user_data) {
  http2_session_data *session_data = (http2_session_data *)user_data;
  (void)session;
  (void)flags;

  switch (frame->hd.type) {
  case NGHTTP2_HEADERS:
    if (frame->headers.cat == NGHTTP2_HCAT_RESPONSE &&
        session_data->stream_data->stream_id == frame->hd.stream_id) {
      /* Print response headers for the initiated request. */
      print_header(stderr, name, namelen, value, valuelen);
      break;
    }
  }
  return 0;
}

/* nghttp2_on_begin_headers_callback: Called when nghttp2 library gets
   started to receive header block. */
static int on_begin_headers_callback(nghttp2_session *session,
                                     const nghttp2_frame *frame,
                                     void *user_data) {
  http2_session_data *session_data = (http2_session_data *)user_data;
  (void)session;

  switch (frame->hd.type) {
  case NGHTTP2_HEADERS:
    if (frame->headers.cat == NGHTTP2_HCAT_RESPONSE &&
        session_data->stream_data->stream_id == frame->hd.stream_id) {
      fprintf(stderr, "Response headers for stream ID=%d:\n",
              frame->hd.stream_id);
    }
    break;
  }
  return 0;
}

/* nghttp2_on_frame_recv_callback: Called when nghttp2 library
   received a complete frame from the remote peer. */
static int on_frame_recv_callback(nghttp2_session *session,
                                  const nghttp2_frame *frame, void *user_data) {
  http2_session_data *session_data = (http2_session_data *)user_data;
  (void)session;

  switch (frame->hd.type) {
  case NGHTTP2_HEADERS:
    if (frame->headers.cat == NGHTTP2_HCAT_RESPONSE &&
        session_data->stream_data->stream_id == frame->hd.stream_id) {
      fprintf(stderr, "All headers received\n");
    }
    break;
  }
  return 0;
}

/* nghttp2_on_data_chunk_recv_callback: Called when DATA frame is
   received from the remote peer. In this implementation, if the frame
   is meant to the stream we initiated, print the received data in
   stdout, so that the user can redirect its output to the file
   easily. */
static int on_data_chunk_recv_callback(nghttp2_session *session, uint8_t flags,
                                       int32_t stream_id, const uint8_t *data,
                                       size_t len, void *user_data) {
  http2_session_data *session_data = (http2_session_data *)user_data;
  (void)session;
  (void)flags;

  if (session_data->stream_data->stream_id == stream_id) {
    fwrite(data, 1, len, stdout);
  }
  return 0;
}

/* nghttp2_on_stream_close_callback: Called when a stream is about to
   closed. This example program only deals with 1 HTTP request (1
   stream), if it is closed, we send GOAWAY and tear down the
   session */
static int on_stream_close_callback(nghttp2_session *session, int32_t stream_id,
                                    uint32_t error_code, void *user_data) {
  http2_session_data *session_data = (http2_session_data *)user_data;
  int rv;

  if (session_data->stream_data->stream_id == stream_id) {
    fprintf(stderr, "Stream %d closed with error_code=%u\n", stream_id,
            error_code);
    rv = nghttp2_session_terminate_session(session, NGHTTP2_NO_ERROR);
    if (rv != 0) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
  }
  return 0;
}

#ifndef OPENSSL_NO_NEXTPROTONEG
/* NPN TLS extension client callback. We check that server advertised
   the HTTP/2 protocol the nghttp2 library supports. If not, exit
   the program. */
static int select_next_proto_cb(SSL *ssl, unsigned char **out,
                                unsigned char *outlen, const unsigned char *in,
                                unsigned int inlen, void *arg) {
  (void)ssl;
  (void)arg;

  if (nghttp2_select_next_protocol(out, outlen, in, inlen) <= 0) {
    errx(1, "Server did not advertise " NGHTTP2_PROTO_VERSION_ID);
  }
  return SSL_TLSEXT_ERR_OK;
}
#endif /* !OPENSSL_NO_NEXTPROTONEG */

/* Create SSL_CTX. */
static SSL_CTX *create_ssl_ctx(void) {
  SSL_CTX *ssl_ctx;
  ssl_ctx = SSL_CTX_new(TLS_client_method());
  if (!ssl_ctx) {
    errx(1, "Could not create SSL/TLS context: %s",
         ERR_error_string(ERR_get_error(), NULL));
  }
  SSL_CTX_set_options(ssl_ctx,
                      SSL_OP_ALL | SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 |
                          SSL_OP_NO_COMPRESSION |
                          SSL_OP_NO_SESSION_RESUMPTION_ON_RENEGOTIATION);
#ifndef OPENSSL_NO_NEXTPROTONEG
  SSL_CTX_set_next_proto_select_cb(ssl_ctx, select_next_proto_cb, NULL);
#endif /* !OPENSSL_NO_NEXTPROTONEG */

#if OPENSSL_VERSION_NUMBER >= 0x10002000L
  SSL_CTX_set_alpn_protos(ssl_ctx, (const unsigned char *)"\x02h2", 3);
#endif /* OPENSSL_VERSION_NUMBER >= 0x10002000L */

  return ssl_ctx;
}

/* Create SSL object */
static SSL *create_ssl(SSL_CTX *ssl_ctx) {
  SSL *ssl;
  ssl = SSL_new(ssl_ctx);
  if (!ssl) {
    errx(1, "Could not create SSL/TLS session object: %s",
         ERR_error_string(ERR_get_error(), NULL));
  }
  return ssl;
}

static void initialize_nghttp2_session(http2_session_data *session_data) {
  nghttp2_session_callbacks *callbacks;

  nghttp2_session_callbacks_new(&callbacks);

  nghttp2_session_callbacks_set_send_callback(callbacks, send_callback);

  nghttp2_session_callbacks_set_on_frame_recv_callback(callbacks,
                                                       on_frame_recv_callback);

  nghttp2_session_callbacks_set_on_data_chunk_recv_callback(
      callbacks, on_data_chunk_recv_callback);

  nghttp2_session_callbacks_set_on_stream_close_callback(
      callbacks, on_stream_close_callback);

  nghttp2_session_callbacks_set_on_header_callback(callbacks,
                                                   on_header_callback);

  nghttp2_session_callbacks_set_on_begin_headers_callback(
      callbacks, on_begin_headers_callback);

  nghttp2_session_client_new(&session_data->session, callbacks, session_data);

  nghttp2_session_callbacks_del(callbacks);
}

static void send_client_connection_header(http2_session_data *session_data) {
  nghttp2_settings_entry iv[1] = {
      {NGHTTP2_SETTINGS_MAX_CONCURRENT_STREAMS, 100}};
  int rv;

  /* client 24 bytes magic string will be sent by nghttp2 library */
  rv = nghttp2_submit_settings(session_data->session, NGHTTP2_FLAG_NONE, iv,
                               ARRLEN(iv));
  if (rv != 0) {
    errx(1, "Could not submit SETTINGS: %s", nghttp2_strerror(rv));
  }
}

#define MAKE_NV(NAME, VALUE, VALUELEN)                                         \
  {                                                                            \
    (uint8_t *)NAME, (uint8_t *)VALUE, sizeof(NAME) - 1, VALUELEN,             \
        NGHTTP2_NV_FLAG_NONE                                                   \
  }

#define MAKE_NV2(NAME, VALUE)                                                  \
  {                                                                            \
    (uint8_t *)NAME, (uint8_t *)VALUE, sizeof(NAME) - 1, sizeof(VALUE) - 1,    \
        NGHTTP2_NV_FLAG_NONE                                                   \
  }

/* Send HTTP request to the remote peer */
static void submit_request(http2_session_data *session_data) {
  int32_t stream_id;
  http2_stream_data *stream_data = session_data->stream_data;
  const char *uri = stream_data->uri;
  const struct http_parser_url *u = stream_data->u;
  nghttp2_nv hdrs[] = {
      MAKE_NV2(":method", "GET"),
      MAKE_NV(":scheme", &uri[u->field_data[UF_SCHEMA].off],
              u->field_data[UF_SCHEMA].len),
      MAKE_NV(":authority", stream_data->authority, stream_data->authoritylen),
      MAKE_NV(":path", stream_data->path, stream_data->pathlen)};
  fprintf(stderr, "Request headers:\n");
  print_headers(stderr, hdrs, ARRLEN(hdrs));
  stream_id = nghttp2_submit_request(session_data->session, NULL, hdrs,
                                     ARRLEN(hdrs), NULL, stream_data);
  if (stream_id < 0) {
    errx(1, "Could not submit HTTP request: %s", nghttp2_strerror(stream_id));
  }

  stream_data->stream_id = stream_id;
}

/* Serialize the frame and send (or buffer) the data to
   bufferevent. */
static int session_send(http2_session_data *session_data) {
  int rv;

  rv = nghttp2_session_send(session_data->session);
  if (rv != 0) {
    warnx("Fatal error: %s", nghttp2_strerror(rv));
    return -1;
  }
  return 0;
}

/* readcb for bufferevent. Here we get the data from the input buffer
   of bufferevent and feed them to nghttp2 library. This may invoke
   nghttp2 callbacks. It may also queues the frame in nghttp2 session
   context. To send them, we call session_send() in the end. */
static void readcb(struct bufferevent *bev, void *ptr) {
  http2_session_data *session_data = (http2_session_data *)ptr;
  ssize_t readlen;
  struct evbuffer *input = bufferevent_get_input(bev);
  size_t datalen = evbuffer_get_length(input);
  unsigned char *data = evbuffer_pullup(input, -1);

  printf("recv %ld\n", datalen);

  readlen = nghttp2_session_mem_recv(session_data->session, data, datalen);
  if (readlen < 0) {
    warnx("Fatal error: %s", nghttp2_strerror((int)readlen));
    delete_http2_session_data(session_data);
    return;
  }
  if (evbuffer_drain(input, (size_t)readlen) != 0) {
    warnx("Fatal error: evbuffer_drain failed");
    delete_http2_session_data(session_data);
    return;
  }
  if (session_send(session_data) != 0) {
    delete_http2_session_data(session_data);
    return;
  }
}

/* writecb for bufferevent. To greaceful shutdown after sending or
   receiving GOAWAY, we check the some conditions on the nghttp2
   library and output buffer of bufferevent. If it indicates we have
   no business to this session, tear down the connection. */
static void writecb(struct bufferevent *bev, void *ptr) {
  http2_session_data *session_data = (http2_session_data *)ptr;
  (void)bev;

  if (nghttp2_session_want_read(session_data->session) == 0 &&
      nghttp2_session_want_write(session_data->session) == 0 &&
      evbuffer_get_length(bufferevent_get_output(session_data->bev)) == 0) {
    delete_http2_session_data(session_data);
  }
}

/* eventcb for bufferevent. For the purpose of simplicity and
   readability of the example program, we omitted the certificate and
   peer verification. After SSL/TLS handshake is over, initialize
   nghttp2 library session, and send client connection header. Then
   send HTTP request. */
static void eventcb(struct bufferevent *bev, short events, void *ptr) {
  http2_session_data *session_data = (http2_session_data *)ptr;
  if (events & BEV_EVENT_CONNECTED) {
    int fd = bufferevent_getfd(bev);
    int val = 1;
    const unsigned char *alpn = NULL;
    unsigned int alpnlen = 0;
    SSL *ssl;

    fprintf(stderr, "Connected\n");

    ssl = bufferevent_openssl_get_ssl(session_data->bev);

#ifndef OPENSSL_NO_NEXTPROTONEG
    SSL_get0_next_proto_negotiated(ssl, &alpn, &alpnlen);
#endif /* !OPENSSL_NO_NEXTPROTONEG */
#if OPENSSL_VERSION_NUMBER >= 0x10002000L
    if (alpn == NULL) {
      SSL_get0_alpn_selected(ssl, &alpn, &alpnlen);
    }
#endif /* OPENSSL_VERSION_NUMBER >= 0x10002000L */

    if (alpn == NULL || alpnlen != 2 || memcmp("h2", alpn, 2) != 0) {
      fprintf(stderr, "h2 is not negotiated\n");
      delete_http2_session_data(session_data);
      return;
    }

    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char *)&val, sizeof(val));
    initialize_nghttp2_session(session_data);
    send_client_connection_header(session_data);
    submit_request(session_data);
    if (session_send(session_data) != 0) {
      delete_http2_session_data(session_data);
    }
    return;
  }
  if (events & BEV_EVENT_EOF) {
    warnx("Disconnected from the remote host");
  } else if (events & BEV_EVENT_ERROR) {
    warnx("Network error");
  } else if (events & BEV_EVENT_TIMEOUT) {
    warnx("Timeout");
  }
  delete_http2_session_data(session_data);
}

/* Start connecting to the remote peer |host:port| */
static void initiate_connection(struct event_base *evbase, SSL_CTX *ssl_ctx,
                                const char *host, uint16_t port,
                                http2_session_data *session_data) {
  int rv;
  struct bufferevent *bev;
  SSL *ssl;

  ssl = create_ssl(ssl_ctx);
  bev = bufferevent_openssl_socket_new(
      evbase, -1, ssl, BUFFEREVENT_SSL_CONNECTING,
      BEV_OPT_DEFER_CALLBACKS | BEV_OPT_CLOSE_ON_FREE);
  bufferevent_enable(bev, EV_READ | EV_WRITE);
  bufferevent_setcb(bev, readcb, writecb, eventcb, session_data);
  rv = bufferevent_socket_connect_hostname(bev, session_data->dnsbase,
                                           AF_UNSPEC, host, port);

  if (rv != 0) {
    errx(1, "Could not connect to the remote host %s", host);
  }
  session_data->bev = bev;
}

/* Get resource denoted by the |uri|. The debug and error messages are
   printed in stderr, while the response body is printed in stdout. */
static void run(const char *uri) {
  struct http_parser_url u;
  char *host;
  uint16_t port;
  int rv;
  SSL_CTX *ssl_ctx;
  struct event_base *evbase;
  http2_session_data *session_data;

  /* Parse the |uri| and stores its components in |u| */
  rv = http_parser_parse_url(uri, strlen(uri), 0, &u);
  if (rv != 0) {
    errx(1, "Could not parse URI %s", uri);
  }
  host = strndup(&uri[u.field_data[UF_HOST].off], u.field_data[UF_HOST].len);
  if (!(u.field_set & (1 << UF_PORT))) {
    port = 443;
  } else {
    port = u.port;
  }

  ssl_ctx = create_ssl_ctx();

  evbase = event_base_new();

  session_data = create_http2_session_data(evbase);
  session_data->stream_data = create_http2_stream_data(uri, &u);

  initiate_connection(evbase, ssl_ctx, host, port, session_data);
  free(host);
  host = NULL;

  event_base_loop(evbase, 0);

  event_base_free(evbase);
  SSL_CTX_free(ssl_ctx);
}

int main(int argc, char **argv) {
  struct sigaction act;

  if (argc < 2) {
    fprintf(stderr, "Usage: libevent-client HTTPS_URI\n");
    exit(EXIT_FAILURE);
  }

  memset(&act, 0, sizeof(struct sigaction));
  act.sa_handler = SIG_IGN;
  sigaction(SIGPIPE, &act, NULL);

#if OPENSSL_VERSION_NUMBER >= 0x1010000fL
  /* No explicit initialization is required. */
#elif defined(OPENSSL_IS_BORINGSSL)
  CRYPTO_library_init();
#else  /* !(OPENSSL_VERSION_NUMBER >= 0x1010000fL) &&                          \
          !defined(OPENSSL_IS_BORINGSSL) */
  OPENSSL_config(NULL);
  SSL_load_error_strings();
  SSL_library_init();
  OpenSSL_add_all_algorithms();
#endif /* !(OPENSSL_VERSION_NUMBER >= 0x1010000fL) &&                          \
          !defined(OPENSSL_IS_BORINGSSL) */

  run(argv[1]);
  return 0;
}

```

这是我在上面的代码上修改的，希望去掉libevent依赖，改为使用 socket fd + select的原始方式，不能正常工作：

```c
/*
 * nghttp2 - HTTP/2 C Library
 *
 * Copyright (c) 2013 Tatsuhiro Tsujikawa
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 * OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#ifdef __sgi
#  include <string.h>
#  define errx(exitcode, format, args...)                                      \
    {                                                                          \
      warnx(format, ##args);                                                   \
      exit(exitcode);                                                          \
    }
#  define warnx(format, args...) fprintf(stderr, format "\n", ##args)
char *strndup(const char *s, size_t size);
#endif

#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif /* HAVE_CONFIG_H */

#include <sys/types.h>
#ifdef HAVE_UNISTD_H
#  include <unistd.h>
#endif /* HAVE_UNISTD_H */
#ifdef HAVE_SYS_SOCKET_H
#  include <sys/socket.h>
#endif /* HAVE_SYS_SOCKET_H */
#ifdef HAVE_NETINET_IN_H
#  include <netinet/in.h>
#endif /* HAVE_NETINET_IN_H */
#include <netinet/tcp.h>
#ifndef __sgi
#  include <err.h>
#endif
#include <signal.h>
#include <string.h>
#include <arpa/inet.h>
#include <fcntl.h>

#include <openssl/ssl.h>
#include <openssl/err.h>
#include <openssl/conf.h>



#include <nghttp2/nghttp2.h>

#include "url_parser.h"

#define ARRLEN(x) (sizeof(x) / sizeof(x[0]))

typedef struct {
  /* The NULL-terminated URI string to retrieve. */
  const char *uri;
  /* Parsed result of the |uri| */
  struct http_parser_url *u;
  /* The authority portion of the |uri|, not NULL-terminated */
  char *authority;
  /* The path portion of the |uri|, including query, not
     NULL-terminated */
  char *path;
  /* The length of the |authority| */
  size_t authoritylen;
  /* The length of the |path| */
  size_t pathlen;
  /* The stream ID of this stream */
  int32_t stream_id;
} http2_stream_data;

typedef struct {
  nghttp2_session *session;
  int fd;
  SSL *ssl;
  http2_stream_data *stream_data;
} http2_session_data;

static http2_stream_data *create_http2_stream_data(const char *uri,
                                                   struct http_parser_url *u) {
  /* MAX 5 digits (max 65535) + 1 ':' + 1 NULL (because of snprintf) */
  size_t extra = 7;
  http2_stream_data *stream_data = malloc(sizeof(http2_stream_data));

  stream_data->uri = uri;
  stream_data->u = u;
  stream_data->stream_id = -1;

  stream_data->authoritylen = u->field_data[UF_HOST].len;
  stream_data->authority = malloc(stream_data->authoritylen + extra);
  memcpy(stream_data->authority, &uri[u->field_data[UF_HOST].off],
         u->field_data[UF_HOST].len);
  if (u->field_set & (1 << UF_PORT)) {
    stream_data->authoritylen +=
        (size_t)snprintf(stream_data->authority + u->field_data[UF_HOST].len,
                         extra, ":%u", u->port);
  }

  /* If we don't have path in URI, we use "/" as path. */
  stream_data->pathlen = 1;
  if (u->field_set & (1 << UF_PATH)) {
    stream_data->pathlen = u->field_data[UF_PATH].len;
  }
  if (u->field_set & (1 << UF_QUERY)) {
    /* +1 for '?' character */
    stream_data->pathlen += (size_t)(u->field_data[UF_QUERY].len + 1);
  }

  stream_data->path = malloc(stream_data->pathlen);
  if (u->field_set & (1 << UF_PATH)) {
    memcpy(stream_data->path, &uri[u->field_data[UF_PATH].off],
           u->field_data[UF_PATH].len);
  } else {
    stream_data->path[0] = '/';
  }
  if (u->field_set & (1 << UF_QUERY)) {
    stream_data->path[stream_data->pathlen - u->field_data[UF_QUERY].len - 1] =
        '?';
    memcpy(stream_data->path + stream_data->pathlen -
               u->field_data[UF_QUERY].len,
           &uri[u->field_data[UF_QUERY].off], u->field_data[UF_QUERY].len);
  }

  return stream_data;
}

static void delete_http2_stream_data(http2_stream_data *stream_data) {
  free(stream_data->path);
  free(stream_data->authority);
  free(stream_data);
}

/* Initializes |session_data| */
static http2_session_data *create_http2_session_data() {
  http2_session_data *session_data = malloc(sizeof(http2_session_data));

  memset(session_data, 0, sizeof(http2_session_data));
  
  return session_data;
}

static void delete_http2_session_data(http2_session_data *session_data) {
  

  if (session_data->ssl) {
    SSL_shutdown(session_data->ssl);
  }
  
  if (session_data->session)
  {
    nghttp2_session_del(session_data->session);
    session_data->session = NULL;
  }
  
  if (session_data->stream_data) {
    delete_http2_stream_data(session_data->stream_data);
    session_data->stream_data = NULL;
  }
  free(session_data);
}

static void print_header(FILE *f, const uint8_t *name, size_t namelen,
                         const uint8_t *value, size_t valuelen) {
  fwrite(name, 1, namelen, f);
  fprintf(f, ": ");
  fwrite(value, 1, valuelen, f);
  fprintf(f, "\n");
}

/* Print HTTP headers to |f|. Please note that this function does not
   take into account that header name and value are sequence of
   octets, therefore they may contain non-printable characters. */
static void print_headers(FILE *f, nghttp2_nv *nva, size_t nvlen) {
  size_t i;
  for (i = 0; i < nvlen; ++i) {
    print_header(f, nva[i].name, nva[i].namelen, nva[i].value, nva[i].valuelen);
  }
  fprintf(f, "\n");
}

/* nghttp2_send_callback. Here we transmit the |data|, |length| bytes,
   to the network. Because we are using libevent bufferevent, we just
   write those bytes into bufferevent buffer. */
static ssize_t send_callback(nghttp2_session *session, const uint8_t *data,
                             size_t length, int flags, void *user_data) {
  http2_session_data *session_data = (http2_session_data *)user_data;
  printf("%s\n", __FUNCTION__);

  return send(session_data->fd, data, length, 0 );
  
}
static ssize_t recv_callback(nghttp2_session *session, uint8_t *buf,
                                         size_t length, int flags,
                                         void *user_data) {
  http2_session_data *session_data = (http2_session_data *)user_data;
  printf("%s\n", __FUNCTION__);
  
}

/* nghttp2_on_header_callback: Called when nghttp2 library emits
   single header name/value pair. */
static int on_header_callback(nghttp2_session *session,
                              const nghttp2_frame *frame, const uint8_t *name,
                              size_t namelen, const uint8_t *value,
                              size_t valuelen, uint8_t flags, void *user_data) {
  http2_session_data *session_data = (http2_session_data *)user_data;
  printf("%s\n", __FUNCTION__);

  switch (frame->hd.type) {
  case NGHTTP2_HEADERS:
    if (frame->headers.cat == NGHTTP2_HCAT_RESPONSE &&
        session_data->stream_data->stream_id == frame->hd.stream_id) {
      /* Print response headers for the initiated request. */
      print_header(stderr, name, namelen, value, valuelen);
      break;
    }
  }
  return 0;
}

/* nghttp2_on_begin_headers_callback: Called when nghttp2 library gets
   started to receive header block. */
static int on_begin_headers_callback(nghttp2_session *session,
                                     const nghttp2_frame *frame,
                                     void *user_data) {
  http2_session_data *session_data = (http2_session_data *)user_data;
  printf("%s\n", __FUNCTION__);

  switch (frame->hd.type) {
  case NGHTTP2_HEADERS:
    if (frame->headers.cat == NGHTTP2_HCAT_RESPONSE &&
        session_data->stream_data->stream_id == frame->hd.stream_id) {
      fprintf(stderr, "Response headers for stream ID=%d:\n",
              frame->hd.stream_id);
    }
    break;
  }
  return 0;
}

/* nghttp2_on_frame_recv_callback: Called when nghttp2 library
   received a complete frame from the remote peer. */
static int on_frame_recv_callback(nghttp2_session *session,
                                  const nghttp2_frame *frame, void *user_data) {
  http2_session_data *session_data = (http2_session_data *)user_data;
  printf("%s\n", __FUNCTION__);

  switch (frame->hd.type) {
  case NGHTTP2_HEADERS:
    if (frame->headers.cat == NGHTTP2_HCAT_RESPONSE &&
        session_data->stream_data->stream_id == frame->hd.stream_id) {
      fprintf(stderr, "All headers received\n");
    }
    break;
  }
  return 0;
}

/* nghttp2_on_data_chunk_recv_callback: Called when DATA frame is
   received from the remote peer. In this implementation, if the frame
   is meant to the stream we initiated, print the received data in
   stdout, so that the user can redirect its output to the file
   easily. */
static int on_data_chunk_recv_callback(nghttp2_session *session, uint8_t flags,
                                       int32_t stream_id, const uint8_t *data,
                                       size_t len, void *user_data) {
  http2_session_data *session_data = (http2_session_data *)user_data;
  printf("%s\n", __FUNCTION__);

  if (session_data->stream_data->stream_id == stream_id) {
    fwrite(data, 1, len, stdout);
  }
  return 0;
}

/* nghttp2_on_stream_close_callback: Called when a stream is about to
   closed. This example program only deals with 1 HTTP request (1
   stream), if it is closed, we send GOAWAY and tear down the
   session */
static int on_stream_close_callback(nghttp2_session *session, int32_t stream_id,
                                    uint32_t error_code, void *user_data) {
  http2_session_data *session_data = (http2_session_data *)user_data;
  int rv;
  printf("%s\n", __FUNCTION__);

  if (session_data->stream_data->stream_id == stream_id) {
    fprintf(stderr, "Stream %d closed with error_code=%u\n", stream_id,
            error_code);
    rv = nghttp2_session_terminate_session(session, NGHTTP2_NO_ERROR);
    if (rv != 0) {
      return NGHTTP2_ERR_CALLBACK_FAILURE;
    }
  }
  return 0;
}

#ifndef OPENSSL_NO_NEXTPROTONEG
/* NPN TLS extension client callback. We check that server advertised
   the HTTP/2 protocol the nghttp2 library supports. If not, exit
   the program. */
static int select_next_proto_cb(SSL *ssl, unsigned char **out,
                                unsigned char *outlen, const unsigned char *in,
                                unsigned int inlen, void *arg) {
  printf("%s\n", __FUNCTION__);

  if (nghttp2_select_next_protocol(out, outlen, in, inlen) <= 0) {
    errx(1, "Server did not advertise " NGHTTP2_PROTO_VERSION_ID);
  }
  return SSL_TLSEXT_ERR_OK;
}
#endif /* !OPENSSL_NO_NEXTPROTONEG */

/* Create SSL_CTX. */
static SSL_CTX *create_ssl_ctx(void) {
  SSL_CTX *ssl_ctx;
  ssl_ctx = SSL_CTX_new(SSLv23_client_method());
  if (!ssl_ctx) {
    errx(1, "Could not create SSL/TLS context: %s",
         ERR_error_string(ERR_get_error(), NULL));
  }
  SSL_CTX_set_options(ssl_ctx,
                      SSL_OP_ALL | SSL_OP_NO_SSLv2 | SSL_OP_NO_SSLv3 |
                          SSL_OP_NO_COMPRESSION |
                          SSL_OP_NO_SESSION_RESUMPTION_ON_RENEGOTIATION);
#ifndef OPENSSL_NO_NEXTPROTONEG
  SSL_CTX_set_next_proto_select_cb(ssl_ctx, select_next_proto_cb, NULL);
#endif /* !OPENSSL_NO_NEXTPROTONEG */

#if OPENSSL_VERSION_NUMBER >= 0x10002000L
  SSL_CTX_set_alpn_protos(ssl_ctx, (const unsigned char *)"\x02h2", 3);
#endif /* OPENSSL_VERSION_NUMBER >= 0x10002000L */

  return ssl_ctx;
}

/* Create SSL object */
static SSL *create_ssl(SSL_CTX *ssl_ctx) {
  SSL *ssl;
  ssl = SSL_new(ssl_ctx);
  if (!ssl) {
    errx(1, "Could not create SSL/TLS session object: %s",
         ERR_error_string(ERR_get_error(), NULL));
  }
  return ssl;
}

static void initialize_nghttp2_session(http2_session_data *session_data) {
  nghttp2_session_callbacks *callbacks;

  nghttp2_session_callbacks_new(&callbacks);

  nghttp2_session_callbacks_set_send_callback(callbacks, send_callback);
  nghttp2_session_callbacks_set_recv_callback(callbacks, recv_callback);

  nghttp2_session_callbacks_set_on_frame_recv_callback(callbacks,
                                                       on_frame_recv_callback);

  nghttp2_session_callbacks_set_on_data_chunk_recv_callback(
      callbacks, on_data_chunk_recv_callback);

  nghttp2_session_callbacks_set_on_stream_close_callback(
      callbacks, on_stream_close_callback);

  nghttp2_session_callbacks_set_on_header_callback(callbacks,
                                                   on_header_callback);

  nghttp2_session_callbacks_set_on_begin_headers_callback(
      callbacks, on_begin_headers_callback);

  nghttp2_session_client_new(&session_data->session, callbacks, session_data);

  nghttp2_session_callbacks_del(callbacks);
}

static void send_client_connection_header(http2_session_data *session_data) {
  nghttp2_settings_entry iv[1] = {
      {NGHTTP2_SETTINGS_MAX_CONCURRENT_STREAMS, 100}};
  int rv;

  /* client 24 bytes magic string will be sent by nghttp2 library */
  rv = nghttp2_submit_settings(session_data->session, NGHTTP2_FLAG_NONE, iv,
                               ARRLEN(iv));
  if (rv != 0) {
    errx(1, "Could not submit SETTINGS: %s", nghttp2_strerror(rv));
  }
}

#define MAKE_NV(NAME, VALUE, VALUELEN)                                         \
  {                                                                            \
    (uint8_t *)NAME, (uint8_t *)VALUE, sizeof(NAME) - 1, VALUELEN,             \
        NGHTTP2_NV_FLAG_NONE                                                   \
  }

#define MAKE_NV2(NAME, VALUE)                                                  \
  {                                                                            \
    (uint8_t *)NAME, (uint8_t *)VALUE, sizeof(NAME) - 1, sizeof(VALUE) - 1,    \
        NGHTTP2_NV_FLAG_NONE                                                   \
  }

/* Send HTTP request to the remote peer */
static void submit_request(http2_session_data *session_data) {
  int32_t stream_id;
  http2_stream_data *stream_data = session_data->stream_data;
  const char *uri = stream_data->uri;
  const struct http_parser_url *u = stream_data->u;
  nghttp2_nv hdrs[] = {
      MAKE_NV2(":method", "GET"),
      MAKE_NV(":scheme", &uri[u->field_data[UF_SCHEMA].off],
              u->field_data[UF_SCHEMA].len),
      MAKE_NV(":authority", stream_data->authority, stream_data->authoritylen),
      MAKE_NV(":path", stream_data->path, stream_data->pathlen)};
  fprintf(stderr, "Request headers:\n");
  print_headers(stderr, hdrs, ARRLEN(hdrs));
  stream_id = nghttp2_submit_request(session_data->session, NULL, hdrs,
                                     ARRLEN(hdrs), NULL, stream_data);
  if (stream_id < 0) {
    errx(1, "Could not submit HTTP request: %s", nghttp2_strerror(stream_id));
  }

  stream_data->stream_id = stream_id;
}

/* Serialize the frame and send (or buffer) the data to
   bufferevent. */
static int session_send(http2_session_data *session_data) {
  int rv;

  rv = nghttp2_session_send(session_data->session);
  if (rv != 0) {
    warnx("Fatal error: %s", nghttp2_strerror(rv));
    return -1;
  }
  return 0;
}

/* readcb for bufferevent. Here we get the data from the input buffer
   of bufferevent and feed them to nghttp2 library. This may invoke
   nghttp2 callbacks. It may also queues the frame in nghttp2 session
   context. To send them, we call session_send() in the end. */
static int readcb(void *ptr) {
  http2_session_data *session_data = (http2_session_data *)ptr;
  ssize_t readlen;
  static unsigned char data[1024*100];

  ssize_t datalen = recv(session_data->fd, data, sizeof(data), 0);
  printf("read %ld bytes\n", datalen);
  if (datalen <= 0) { 
    delete_http2_session_data(session_data);

    
    return -1;
 }
 

  readlen = nghttp2_session_mem_recv(session_data->session, data, datalen);
  if (readlen < 0) {
    warnx("Fatal error: %s", nghttp2_strerror((int)readlen));
    delete_http2_session_data(session_data);
    return -1;
  }
  printf(">>read %ld bytes\n", readlen);
  
  
  if (session_send(session_data) != 0) {
    delete_http2_session_data(session_data);
    return -1;
  }
  return 0;
}

/* writecb for bufferevent. To greaceful shutdown after sending or
   receiving GOAWAY, we check the some conditions on the nghttp2
   library and output buffer of bufferevent. If it indicates we have
   no business to this session, tear down the connection. */
static void writecb( void *ptr) {
    /*
  http2_session_data *session_data = (http2_session_data *)ptr;

  if (nghttp2_session_want_read(session_data->session) == 0 &&
      nghttp2_session_want_write(session_data->session) == 0 ) {
    delete_http2_session_data(session_data);
  }
  */
}

/* eventcb for bufferevent. For the purpose of simplicity and
   readability of the example program, we omitted the certificate and
   peer verification. After SSL/TLS handshake is over, initialize
   nghttp2 library session, and send client connection header. Then
   send HTTP request. */
static int after_connected(void *ptr) {
  http2_session_data *session_data = (http2_session_data *)ptr;
  
    int val = 1;
    const unsigned char *alpn = NULL;
    unsigned int alpnlen = 0;

    fprintf(stderr, "Connected\n");


#ifndef OPENSSL_NO_NEXTPROTONEG
    SSL_get0_next_proto_negotiated(session_data->ssl, &alpn, &alpnlen);
#endif /* !OPENSSL_NO_NEXTPROTONEG */
#if OPENSSL_VERSION_NUMBER >= 0x10002000L
    if (alpn == NULL) {
      SSL_get0_alpn_selected(session_data->ssl, &alpn, &alpnlen);
    }
#endif /* OPENSSL_VERSION_NUMBER >= 0x10002000L */

    if (alpn == NULL || alpnlen != 2 || memcmp("h2", alpn, 2) != 0) {
      fprintf(stderr, "h2 is not negotiated\n");
      delete_http2_session_data(session_data);
      return -1;
    }

    setsockopt(session_data->fd, IPPROTO_TCP, TCP_NODELAY, (char *)&val, sizeof(val));
    initialize_nghttp2_session(session_data);
    send_client_connection_header(session_data);
    submit_request(session_data);
    if (session_send(session_data) != 0) {
      delete_http2_session_data(session_data);
      return -1;
    }
    return 0;
}

/* Start connecting to the remote peer |host:port| */
static void run(const char * uri, const char *ip, uint16_t port) 
{
  struct http_parser_url u;
  char *host;
  int rv;
  SSL_CTX *ssl_ctx;
  http2_session_data *session_data;

  /* Parse the |uri| and stores its components in |u| */
  rv = http_parser_parse_url(uri, strlen(uri), 0, &u);
  if (rv != 0) {
    errx(1, "Could not parse URI %s", uri);
  }
  host = strndup(&uri[u.field_data[UF_HOST].off], u.field_data[UF_HOST].len);
  if (!(u.field_set & (1 << UF_PORT))) {
    port = 443;
  } else {
    port = u.port;
  }

  ssl_ctx = create_ssl_ctx();
  session_data = create_http2_session_data();
  session_data->stream_data = create_http2_stream_data(uri, &u);
  fd_set readfds, writefds;

  struct sockaddr_in addr = {0};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = inet_addr(ip);
  addr.sin_port = htons(port);
 
  session_data->ssl = create_ssl(ssl_ctx);
  session_data->fd = socket(AF_INET, SOCK_STREAM, 0);
  connect(session_data->fd, (struct sockaddr*)&addr, sizeof(addr));
  SSL_set_fd(session_data->ssl, session_data->fd);
  SSL_connect(session_data->ssl);
 
 
  if (after_connected(session_data) != 0)
    return;

  while (1)
  {
    FD_ZERO(&readfds);
    FD_SET(session_data->fd, &readfds);
    FD_ZERO(&writefds);
    FD_SET(session_data->fd, &writefds);
    struct timeval t;
    t.tv_sec = 1;
    t.tv_usec = 0;

    int fdnum = select(session_data->fd + 1, &readfds, &writefds, NULL, &t);
    if (fdnum <= 0)
    {
        continue;
    }
    if (FD_ISSET(session_data->fd, &readfds))
    {
        
        if (readcb(session_data) <0)
        {
            break;
        }
    }
    if (FD_ISSET(session_data->fd, &writefds))
    {
        writecb(session_data);
    }

  }

  
 
}

int main(int argc, char **argv) {
  struct sigaction act;

  if (argc < 4) {
    fprintf(stderr, "Usage: libevent-client uri ip port\n");
    exit(EXIT_FAILURE);
  }

  memset(&act, 0, sizeof(struct sigaction));
  act.sa_handler = SIG_IGN;
  sigaction(SIGPIPE, &act, NULL);

#if OPENSSL_VERSION_NUMBER >= 0x1010000fL
  /* No explicit initialization is required. */
#elif defined(OPENSSL_IS_BORINGSSL)
  CRYPTO_library_init();
#else  /* !(OPENSSL_VERSION_NUMBER >= 0x1010000fL) &&                          \
          !defined(OPENSSL_IS_BORINGSSL) */
  OPENSSL_config(NULL);
  SSL_load_error_strings();
  SSL_library_init();
  OpenSSL_add_all_algorithms();
#endif /* !(OPENSSL_VERSION_NUMBER >= 0x1010000fL) &&                          \
          !defined(OPENSSL_IS_BORINGSSL) */

  run(argv[1], argv[2], atoi(argv[3]));
  return 0;
}
```

下面是编译上述两个程序的makefile，需要在机器上安装openssl3.x

```makefile
CFLAGS	= -I /usr/local/openssl/include/


all:client ecli

client:client.o url_parser.o
	gcc -g -o client client.o url_parser.o    /usr/local/openssl/lib64/libssl.a /usr/local/openssl/lib64/libcrypto.a /usr/lib/x86_64-linux-gnu/libnghttp2.a -ldl -lpthread

ecli:event_cli.o url_parser.o
	gcc -g -o ecli event_cli.o url_parser.o   /usr/lib/x86_64-linux-gnu/libevent_openssl.a /usr/lib/x86_64-linux-gnu/libevent.a   /usr/local/openssl/lib64/libssl.a /usr/local/openssl/lib64/libcrypto.a /usr/lib/x86_64-linux-gnu/libnghttp2.a  -ldl -lpthread

client.o:client.c
	gcc $(CFLAGS) -g -c client.c 

url_parser.o:url_parser.c
	gcc $(CFLAGS) -g -c url_parser.c

event_cli.o:event_cli.c
	gcc $(CFLAGS) -g -c event_cli.c 

```

测试：

```shell
./client https://www.facebook.com 31.13.75.35 443

 ./ecli  https://www.facebook.com
```

