### 1、基本概念

1. RTCPeerConnection：代表与对端的webrtc连接，最重要的变量，两个peer的代码围绕它玩
2. 信令服务器：作为中间人，帮助两个peer进行相互发现和信令间的交互。
3. ICE：Interactive connectivity Establishment。有点模糊，可以指代浏览器的webrtc内核P2P通信框架
4. STUN：帮助位于NAT局域网内的Peer发现自己的外网地址，从而穿越NAT进行P2P通信
5. NAT：就是NAT咯，有多种类型的NAT，不展开
6. TURN：中转服务器，如果不能直连，就要通过TURN中转。 （不确定） 
7. SDP：会话描述协议，表示参与通信的Peer的能力，例如分辨率、编解码格式、加解密算法等，字段很多但看不明白。通信过程中一方发出offer，一方回以answer。offer和answer里主要构成就是SDP
8. candidate：P2P的外网地址等关乎P2P通信的信息，相当于告诉对方门牌号。即使P2P通信已经建立了，也可能继续交换candidate，并修改P2P通信通道。

更详细准确的描述见下面这个又科普又准确的资料：

```
https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API/Protocols
```




### 2、交互时序

分两部分：

1. 通过STUN发现自己的外网IP和端口，打洞。似乎candidate这个会通过信令服务器转发给对方，不是外网地址发现的应答
2. 通过信令服务器进行相互发现和握手，又包括：
   1. 登录信令服务器，以注册身份
   2. 发送offer，对方会应答以answer，offer和answer两个报文的主要内容就是SDP
   3. 相互发送candidate，会作死的发，一开始发内网地址，等发现了自己的外网地址后就作死的发外网地址。

借网友一张图：

![这里由张图片](img/webrtc/interaction.png)

### 3、实操整起

#### 3.1 先整一个页面，带有js

借网友的代码如下：

```javascript
// filename: rtc.js
  var connectedUser; 
  var myName; //我方用户名
  var yourConnection; 

  // 与信令服务器的websocket连接
  var connection = new WebSocket("ws://119.28.214.71:8090/");
  connection.onopen = function () {
    console.log("Connected");
  };

// 与信令服务器交互的消息回调
  connection.onmessage = function (message) {
    console.log("Got message", message.data);
    var data = JSON.parse(message.data);
    switch (data.type) {
      case "login":
        onLogin(data.success);
        break;
      case "offer":
        onOffer(data.offer, data.name);
        break;
      case "answer":
        onAnswer(data.answer);
        break;
      case "candidate":
        onCandidate(data.candidate);
        break;
      case "leave":
        onLeave();
        break;
      default:
        console.log("default message");
        console.log(data);
        break;
    }
  };

  connection.onerror = function (err) {
    console.log("Got error", err);
  };

// 向信令服务器发送消息
  function send(message) {
    if (connectedUser) {
      //message.name = connectedUser;
      //message.myName = name;
      message.name = theirUsernameInput.value;
      message.myName = usernameInput.value;
    }

    connection.send(JSON.stringify(message));
  };

  // 绑定HTML上的一些标签
  var loginPage = document.querySelector('#login-page'),
      usernameInput = document.querySelector('#username'),
      loginButton = document.querySelector('#login'),
      callPage = document.querySelector('#call-page'),
      theirUsernameInput = document.querySelector('#their-username'),
      callButton = document.querySelector('#call'),
      hangUpButton = document.querySelector('#hang-up'),
      messageInput = document.querySelector('#message'),
      sendButton = document.querySelector('#send'),
      received = document.querySelector('#received');

 // callPage.style.display = "none";

//  登录按钮click事件响应
// Login when the user clicks the button
//  记录登录名，向信令服务器发送登录信息
  loginButton.addEventListener("click", function (event) {
  
    myName = usernameInput.value;

    if (myName.length > 0) {
      send({
        type: "login",
        name: myName
      });
    }
  });

  // 响应信令服务器反馈的登录信息
  function onLogin(success) {
    if (success === false) {
      alert("Login unsuccessful, please try a different name.");
    } else {
      //loginPage.style.display = "none";
      //callPage.style.display = "block";

      // Get the plumbing ready for a call
      //  准备开始一个连接
      startConnection();
    }
  };


  var yourVideo = document.querySelector('#yours'),
      theirVideo = document.querySelector('#theirs'),
      // yourConnection, connectedUser, stream, dataChannel;
      connectedUser, stream, dataChannel;

  // 打开自己的摄像头
  // 准备开始一次peer to peer 连接
  function startConnection() {

    // 想要获取一个最接近 1280x720 的相机分辨率
    var constraints = {audio: false, video: {width: 320, height: 480}};
    navigator.mediaDevices.getUserMedia(constraints)
        .then(function (mediaStream) {
          // var video = document.querySelector('video');

          yourVideo.srcObject = mediaStream;

          if (hasRTCPeerConnection()) {
            console.log("setupPeerConnection .. ")
            setupPeerConnection(mediaStream);
          } else {
            alert("Sorry, your browser does not support WebRTC.");
          }

          yourVideo.onloadedmetadata = function (e) {
            yourVideo.play();
          };


        })
        .catch(function (err) {
          console.log(err.name + " -- : " + err.message);
        });

  }

  // 创建RTCPeerConnection对象 ，绑定ICE服务器，绑定多媒体数据流
  function setupPeerConnection(stream) {
    if (yourConnection == null) {
      var configuration = {
        // "iceServers": [{ "url": "stun:127.0.0.1:9876" }]
        "iceServers": [{"url": "stun:119.28.214.71:3478"}, {
          "url": "turn:119.28.214.71:3478",
          "username": "bison",
          "credential": "liao"
        }]
      };
      yourConnection = new RTCPeerConnection(configuration, {optional: [{RtpDataChannels: true}]});
    }


    if (yourConnection == null) {
      console.log("yourConneion is null");
    } else {
      console.log("yourConnection is a object")
    }

    console.log("========================= setupPeerConnection stream ====================================")
    // console.log(stream);

    // Setup stream listening
    yourConnection.addStream(stream);
    yourConnection.onaddstream = function (e) {

      console.log(e);
      // theirVideo.src = window.URL.createObjectURL(e.stream);
      theirVideo.srcObject = e.stream;
      theirVideo.play();
    };

    // Setup ice handling
    yourConnection.onicecandidate = function (event) {
      if (event.candidate) {
        send({
          type: "candidate",
          candidate: event.candidate
        });
      }
    };

    // 打开数据通道 （这个是用于 文字交流用）
    openDataChannel();
  }

  function openDataChannel() {
    
        dataChannel = yourConnection.createDataChannel("myLabel"); //主动call的一方会直接用dataChannel变量
        yourConnection.ondatachannel = receiveChannelCallback; //被动call的一方会收到这个事件，重新设置dataChannel变量

        dataChannel.onerror = function (error) {
          console.log("Data Channel Error:", error);
        };

        dataChannel.onmessage = function (event) {
          console.log("Got Data Channel Message:", event.data);

          received.innerHTML += event.data + "<br />";
          received.scrollTop = received.scrollHeight;
        };

        dataChannel.onopen = function () {
          dataChannel.send("datachannel has connected.");
        };

        dataChannel.onclose = function () {
          console.log("The Data Channel is Closed");
        };
  }

  function receiveChannelCallback(event) {
    dataChannel = event.channel;
    dataChannel.onmessage = function (event) {
      console.log("Got Data Channel Message:", event.data);

      received.innerHTML += event.data + "<br />";
      received.scrollTop = received.scrollHeight;
    };
    dataChannel.onopen =  function () {
      dataChannel.send("datachannel has connected.");
    };
    dataChannel.onclose = function () {
      console.log("The Data Channel is Closed");
    };
  }


// Bind our text input and received area
  sendButton.addEventListener("click", function (event) {
    var val = messageInput.value;
    received.innerHTML += val + "<br />";
    received.scrollTop = received.scrollHeight;
    dataChannel.send(val);
  });

/*  function hasUserMedia() {
    navigator.getUserMedia = navigator.getUserMedia || navigator.webkitGetUserMedia || navigator.mozGetUserMedia || navigator.msGetUserMedia;
    return !!navigator.getUserMedia;
  }*/

  function hasRTCPeerConnection() {
    window.RTCPeerConnection = window.RTCPeerConnection || window.webkitRTCPeerConnection || window.mozRTCPeerConnection;
    window.RTCSessionDescription = window.RTCSessionDescription || window.webkitRTCSessionDescription || window.mozRTCSessionDescription;
    window.RTCIceCandidate = window.RTCIceCandidate || window.webkitRTCIceCandidate || window.mozRTCIceCandidate;
    return !!window.RTCPeerConnection;
  }

  callButton.addEventListener("click", function () {
    var theirUsername = theirUsernameInput.value;
    console.log("call " + theirUsername)
    if (theirUsername.length > 0) {
      startPeerConnection(theirUsername);
    }
  });

  // 开始peer to peer 连接
  function startPeerConnection(theiruser) {
        connectedUser = theiruser;

        // yourConnection
        // Begin the offer

        // 发送通话请求 1
        yourConnection.createOffer(function (offer) {
          console.log("    yourConnection.createOffer");
          send({
            type: "offer",
            offer: offer
          });

          console.log("     yourConnection.setLocalDescription(offer);");
          yourConnection.setLocalDescription(offer);
        }, function (error) {
          alert("An error has occurred.");
        });
  };

  // 接受通话者 响应 通话请求 2
  function onOffer(offer, name) {
    connectedUser = name;

    console.log("============================================================");
    console.log("===============    onOffer       (===================");
    console.log("connector user name is "+connectedUser);
    console.log("============================================================");


    var offerJson = JSON.parse(offer);
    var sdp = offerJson.sdp;

    //   设置对方的会话描述
    try {
      console.log("                   yourConnection.setRemoteDescription                   ");
      yourConnection.setRemoteDescription(new window.RTCSessionDescription(offerJson), function () {
            console.log("success");
          }
          ,
          function () {
            console.log("fail")
          });

    } catch (e) {
      alert(e)
    }

    // 向通话请求者 发送回复消息 3
    yourConnection.createAnswer(function (answer) {
      yourConnection.setLocalDescription(answer);
      console.log("               yourConnection.createAnswer                  ");
      send({
        type: "answer",
        answer: answer
      });
    }, function (error) {
      alert("An error has occurred");
    });

    console.log("onOffer is success");

  };

  // 通话请求者 处理 回复 4
  function onAnswer(answer) {
    if (yourConnection == null) {
      alert("yourconnection is null in onAnswer");
    }

    console.log("============================================================");
    console.log("================ OnAnswer ============================");
    console.log("============================================================");
    console.log(answer);
    if (answer != null) {
      console.log(typeof answer);
    }

    var answerJson = JSON.parse(answer);
    console.log(answerJson);

    try {

      //  设置本次会话的描述
      yourConnection.setRemoteDescription(new RTCSessionDescription(answerJson));
    } catch (e) {
      alert(e);
    }

    console.log("onAnswer is success");

  };

  // 对ICE候选连接的事情响应
  function onCandidate(candidate) {
    console.log("============================================================");
    console.log("================ OnCandidate ============================");
    console.log("============================================================");
    console.log(candidate);
    if (candidate != null) {
      console.log(typeof candidate);
    }

    var iceCandidate;

    // try {

    var candidateJson = JSON.parse(candidate);
    console.log(candidateJson);

    iceCandidate = new RTCIceCandidate(candidateJson);
    // }catch(e){
    //   console.log("exception is ")
    //   console.log(e);
    // }

    if (yourConnection == null) {
      alert("yourconnection is null in onCandidate");
    }
    // yourConnection.addIceCandidate(new RTCIceCandidate(candidate));
    yourConnection.addIceCandidate(iceCandidate);
  };

  hangUpButton.addEventListener("click", function () {
    send({
      type: "leave"
    });

    onLeave();
  });

  function onLeave() {
    connectedUser = null;
    theirVideo.src = null;
    yourConnection.close();
    yourConnection.onicecandidate = null;
    yourConnection.onaddstream = null;
    setupPeerConnection(stream);
  };

```

```html
<!--
filename:index.html
-->

<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />

    <title>WebRTC</title>
   
    
    <style>
      body {
        background-color: #3D6DF2;
        margin-top: 15px;
        font-family: sans-serif;
        color: white;
      }

      video {
        background: rgb(95, 94, 94);
        border: 1px solid gray;
      }

      .page {
        position: relative;
        display: block;
        margin: 0 auto;
        width: 500px;
        
      }


      #yours {
        width: 150px;
        height: 150px;
        position: absolute;
        top: 15px;
        right: 15px;
      }

      #theirs {
        width: 500px;
        height: 500px;
     
      }

      #received {
        display: block;
        width: 480px;
        height: 100px;
        background: white;
        padding: 10px;
        margin-top: 10px;
        color: black;
        overflow: scroll;
      }
    </style>
   
  </head>
  <body>
    <div id="login-page" class="page">
      <h2>Login As</h2>
      <input type="text" id="username" />
      <button id="login">Login</button>
    </div>
    <br><br>
    <div id="call-page" class="page">
      <video id="yours" muted="muted" autoplay ></video>
      <video id="theirs" muted="muted" autoplay></video>
      peer: <input type="text" id="their-username" />
      <button id="call">Call</button>
      <button id="hang-up">Hang Up</button>
        <br>
      msg :<input type="text" id="message"></input>
      <button id="send">Send</button>
      <br><br>
      <div id="received"></div>
    </div>

   
    <script src=".\rtc.js"></script>
  </body>
</html>

```

html页面很简单，js代码有点复杂，尤其是对于不做前端开发的我。但关键的就这几个函数和步骤：

1. 建立与信令服务器的链接，注册各种回调函数，最重要的就是消息处理函数了
2. 双方都点击login按钮会触发login请求，应答会在上述回调函数的onLogin中处理，onLogin会调用startConnection为P2P连接做准备，重要的两个准备：
   1. navigator.mediaDevices.getUserMedia() 从本地媒体设备获取媒体数据
   2. 请求STUN服务器，生成RTCPeerConnection
3. 一方点击call按钮，会创建和发送offer，同时设置本地的SDP。
4. 对方收到后会在消息回调函数onOffer中处理，会据此设置对端的SDP。并进一步触发创建和发送answer，并设置本地的SDP
5. 发起会话的一方，收到对方的answer后，会据此设置远端的SDP信息。到目前为止，双方都有了自己的SDP和对端的SDP，据此判断能否协商成功
6. 如果上面的SDP协商成功（也可能不成功不是，看不对眼），浏览器会触发注册的 JS 回调代码发送candidate给对方，可能会发送多次；双方会相互发送，收到对方的candidate后会调用 RTCPeerConnection.addIceCandidate()把收到的candidate设置进去，浏览器底层开始P2P通信。

借网友一张图：

![](img/webrtc/SetSDP.png)

所以关键就是在合适的交互环节调用这几个函数：

```javascript
yourConnection = new RTCPeerConnection(...）
yourConnection.addStream(stream)
yourConnection.createDataChannel() // 如果用的话
yourConnection.createOffer()
yourConnection.setLocalDescription(offer)//发出己方的offer或者answer的时候，设置一下本地的SDP
yourConnection.setRemoteDescription()//收到对方的offer或者answer的时候
yourConnection.createAnswer()
yourConnection.addIceCandidate() //收到对方的candidate后，设置进去
同时，yourConnection也支持设置各种onicecandidate()  、onXXX()等事件
```

这个页面里的两个图片也说的很清楚：

```
https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API/Signaling_and_video_calling
```



#### 3.2 搞一个信令服务器

用java的websocket库比较容易搞定：

```java
package io.bison;

import com.alibaba.fastjson.JSONObject;
import java.net.InetSocketAddress;
import java.util.HashMap;
import java.util.Iterator;
import jdk.nashorn.api.scripting.JSObject;
import org.java_websocket.WebSocket;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;


/**
 * websocket服务器
 */
public class SignalServer extends WebSocketServer{
    
    private static final java.util.logging.Logger log = java.util.logging.Logger.getLogger(SignalServer.class.getName());
    private static java.util.Map sessionMap = new java.util.HashMap<String, WebSocket>();
    private static java.util.Map sessionProduce = new HashMap<String, JSObject>();


    public SignalServer(int port) {
        super(new InetSocketAddress(port));
    }
    /**
     * WebSocket连接关闭时调用
     */
    @Override
    public void onClose(WebSocket ws, int arg1, String arg2, boolean arg3) {
        System.out.println("------------------onClose-------------------");
    }

    /**
     * 错误发生时调用。
     */
    @Override
    public void onError(WebSocket ws, Exception e) {
        System.out.println("------------------onError-------------------");
        if(ws != null) {
        }
        e.printStackTrace();
    }

    /**
     * 接收到的消息
     */
    @Override
    public void onMessage(WebSocket ws, String msg) {
        System.out.println("收到消息："+msg);

        
        JSONObject msgJSON = null;
        try
        {
           msgJSON  = JSONObject.parseObject(msg);
        } 
        catch (com.alibaba.fastjson.JSONException e)
        {
            msgJSON = null;
        }

        if (msgJSON == null)
        {
            ws.send("send:"+msg);
        }
        else
        {
            String type = msgJSON.getString("type");
            if (type == null) { type = "";}
            switch (type) {
                case "login": {
       
                    log.info("user login in as " + msgJSON.getString("name")+"\n");
                    if (sessionMap.containsKey(msgJSON.getString("name"))) {
                        sessionMap.put(msgJSON.getString("name"), ws);
                        JSONObject login = new JSONObject();
                        login.put("type", "login");
                        login.put("success", true);
                        ws.send(login.toJSONString());

                    } else {
                        sessionMap.put(msgJSON.getString("name"), ws);
                        JSONObject login = new JSONObject();
                        login.put("type", "login");
                        login.put("success", true);
                        login.put("myName", msgJSON.getString("name"));
                        ws.send(login.toJSONString());
                    }

                }
                break;
                case "offer": {
                    /**
                     * 处理offer消息 offer是一个peer to peer 连接中的 第一步 这个是响应通话发起者的消息
                     * 这里主要是找到 通话发起者要通话的对方的会话
                     */
//                    onOffer(data.offer, data.name);\
                    log.info("Sending offer to " + msgJSON.getString("name") + " from " + msgJSON.getString("myName")+"\n");

                    WebSocket conn = (WebSocket) sessionMap.get(msgJSON.getString("name"));

                    if (conn != null) {
                        JSONObject offer = new JSONObject();
                        offer.put("type", "offer");
                        offer.put("offer", msgJSON.getString("offer"));
                        offer.put("name", msgJSON.getString("name"));
                        conn.send(offer.toJSONString());

                        /**
                         * 保存会话状态
                         */
                        JSONObject offerAnswer = new JSONObject();
                        offerAnswer.put("offerName", msgJSON.getString("myName")); // 222 
                        offerAnswer.put("answerName", msgJSON.getString("name")); // 111
                        sessionProduce.put(msgJSON.getString("name"), offerAnswer); // 111
                    }

                }
                break;
                case "answer": {

                    log.info("answer from " + msgJSON.getString("myName")); //111

                    

                    JSONObject offerAnswer = (JSONObject) sessionProduce.get(msgJSON.getString("myName"));
                    String offerName = offerAnswer.getString("offerName"); // 222
                    WebSocket conn = (WebSocket) sessionMap.get(offerName);
                    if (conn != null) {
                        sessionProduce.remove(msgJSON.getString("myName"));
                        JSONObject answer = new JSONObject();
                        answer.put("type", "answer");
                        answer.put("answer", msgJSON.getString("answer"));
                        conn.send(answer.toJSONString());

                        log.info("send answer to "+offerName+"\n");
                    }
                    else
                    {
                        log.info("faile to find conn for "+offerName+"\n");
                    }

                }
                break;
                case "candidate": {
                    log.info("Sending candidate to " + msgJSON.getString("name"));
                    WebSocket conn = (WebSocket) sessionMap.get(msgJSON.getString("name"));
                    if (conn != null) {
                        JSONObject candidate = new JSONObject();
                        candidate.put("type", "candidate");
                        candidate.put("candidate", msgJSON.getString("candidate"));
                        conn.send(candidate.toJSONString());
                    }
                    else
                    {
                        log.info("failed to find conn for "+msgJSON.getString("name"));
                    }
                }
                break;
                case "leave": {
                    log.info("Disconnectiong user from " + msgJSON.getString("name"));
                    WebSocket conn = (WebSocket) sessionMap.get(msgJSON.getString("name"));

                    if (conn != null) {
                        JSONObject leave = new JSONObject();
                        leave.put("type", "leave");

                        conn.send(leave.toJSONString());
                    }
                }

                break;
                default:
                    JSONObject defaultMsg = new JSONObject();
                    defaultMsg.put("type", "error");
                    defaultMsg.put("message", "Unreconfized command : " + msgJSON.getString("type"));
                    ws.send(defaultMsg.toJSONString());
                    break;
            }
        }

        
        if(ws.isClosed()) {

        } 
        else if (ws.isClosing()) {
            
        } 
        else if(ws.isOpen()) {
            
        }
    }

    /**
     * websocket进行握手之后调用，并且给WebSocket写做准备
     * 通过握手可以获取请求头信息
     */
    @Override
    public void onOpen(WebSocket ws, ClientHandshake shake) {
        System.out.println("-----------------onOpen--------------------"+ws.isOpen()+"--"+ws.getReadyState()+"--"+ws.getAttachment());
        for(Iterator<String> it=shake.iterateHttpFields();it.hasNext();) {
            String key = it.next();
            System.out.println(key+":"+shake.getFieldValue(key));
        }
        JSONObject open = new JSONObject();
        open.put("status", "success");
        ws.send(open.toJSONString());
    }
    /**
     * 当服务器成功启动时调用
     */
    @Override
    public void onStart() {
        System.out.println("------------------onStart-------------------");
    }
}
```

```java
package io.bison;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        new SignalServer(8090).start();
        System.out.println( "Hello World!" );
    }
}
```

我用的是maven，具体怎么创建一个工程，可以参考java grpc那一篇

这里有个插曲，因为要操作香港的信令服务器，所以用了vpn，本地的浏览器访问香港信令服务器ws协议走不通，关闭vpn就好了。

到这一步的时候，同局域网的两个浏览器已经可以视频了，不需要STUN服务器。

#### 3.3 搞一个STUN服务器

```shell
sudo apt install coturn

# 修改一下/etc/turnserver.conf里面的用户密码：user=bison:liao
sudo service coturn start
```

### 4、看看通信的报文

![](img/webrtc/example.png)

#### 4.1 websocket握手

```
Accept-Encoding:gzip, deflate
Accept-Language:en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7
Cache-Control:no-cache
Connection:Upgrade
Host:119.28.214.71:8090
Origin:null
Pragma:no-cache
Sec-WebSocket-Extensions:permessage-deflate; client_max_window_bits
Sec-WebSocket-Key:I3vjXv/JgSrTqesfM5QG9Q==
Sec-WebSocket-Version:13
Upgrade:websocket
User-Agent:Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36 Edg/107.0.1418.42
```

#### 4.2 两方login

```
收到消息：{"type":"login","name":"111"}
收到消息：{"type":"login","name":"222"}
```

#### 4.3 一方进行call，刷的就视频接通了

```shell
收到消息：{"type":"offer","offer":{"sdp":"v=0...2"}
Nov 12, 2022 8:23:14 PM io.bison.SignalServer onMessage
INFO: Sending offer to 111 from 222

SDP：
v=0
o=- 727800857132210897 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0 1
a=extmap-allow-mixed
a=msid-semantic: WMS 5FBEHloS5pG7M3cjTBA8kGFpDcT8gP5CBxif
m=video 9 UDP/TLS/RTP/SAVPF 96 97 102 122 127 121 125 107 108 109 124 120 39 40 98 99 100 101 123 119 114 115 116
c=IN IP4 0.0.0.0
a=rtcp:9 IN IP4 0.0.0.0
a=ice-ufrag:RUtG
a=ice-pwd:aMpyBrT2rIfKL8JM+9Q6puft
a=ice-options:trickle
a=fingerprint:sha-256 D7:AC:4D:97:15:C7:17:56:4C:0E:F4:44:CF:0B:EA:E7:32:EF:25:1B:61:83:26:37:DD:52:9F:F5:B9:97:67:FA
a=setup:actpass
a=mid:0
a=extmap:1 urn:ietf:params:rtp-hdrext:toffset
a=extmap:2 http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time
a=extmap:3 urn:3gpp:video-orientation
a=extmap:4 http://www.ietf.org/id/draft-holmer-rmcat-transport-wide-cc-extensions-01
a=extmap:5 http://www.webrtc.org/experiments/rtp-hdrext/playout-delay
a=extmap:6 http://www.webrtc.org/experiments/rtp-hdrext/video-content-type
a=extmap:7 http://www.webrtc.org/experiments/rtp-hdrext/video-timing
a=extmap:8 http://www.webrtc.org/experiments/rtp-hdrext/color-space
a=extmap:9 urn:ietf:params:rtp-hdrext:sdes:mid
a=extmap:10 urn:ietf:params:rtp-hdrext:sdes:rtp-stream-id
a=extmap:11 urn:ietf:params:rtp-hdrext:sdes:repaired-rtp-stream-id
a=sendrecv
a=msid:5FBEHloS5pG7M3cjTBA8kGFpDcT8gP5CBxif e2932a59-9078-4795-8923-41a8ba7c8b62
a=rtcp-mux
a=rtcp-rsize
a=rtpmap:96 VP8/90000
a=rtcp-fb:96 goog-remb
a=rtcp-fb:96 transport-cc
a=rtcp-fb:96 ccm fir
a=rtcp-fb:96 nack
a=rtcp-fb:96 nack pli
a=rtpmap:97 rtx/90000
a=fmtp:97 apt=96
a=rtpmap:102 H264/90000
a=rtcp-fb:102 goog-remb
a=rtcp-fb:102 transport-cc
a=rtcp-fb:102 ccm fir
a=rtcp-fb:102 nack
a=rtcp-fb:102 nack pli
a=fmtp:102 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f
a=rtpmap:122 rtx/90000
a=fmtp:122 apt=102
a=rtpmap:127 H264/90000
a=rtcp-fb:127 goog-remb
a=rtcp-fb:127 transport-cc
a=rtcp-fb:127 ccm fir
a=rtcp-fb:127 nack
a=rtcp-fb:127 nack pli
a=fmtp:127 level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=42001f
a=rtpmap:121 rtx/90000
a=fmtp:121 apt=127
a=rtpmap:125 H264/90000
a=rtcp-fb:125 goog-remb
a=rtcp-fb:125 transport-cc
a=rtcp-fb:125 ccm fir
a=rtcp-fb:125 nack
a=rtcp-fb:125 nack pli
a=fmtp:125 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f
a=rtpmap:107 rtx/90000
a=fmtp:107 apt=125
a=rtpmap:108 H264/90000
a=rtcp-fb:108 goog-remb
a=rtcp-fb:108 transport-cc
a=rtcp-fb:108 ccm fir
a=rtcp-fb:108 nack
a=rtcp-fb:108 nack pli
a=fmtp:108 level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=42e01f
a=rtpmap:109 rtx/90000
a=fmtp:109 apt=108
a=rtpmap:124 H264/90000
a=rtcp-fb:124 goog-remb
a=rtcp-fb:124 transport-cc
a=rtcp-fb:124 ccm fir
a=rtcp-fb:124 nack
a=rtcp-fb:124 nack pli
a=fmtp:124 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=4d001f
a=rtpmap:120 rtx/90000
a=fmtp:120 apt=124
a=rtpmap:39 H264/90000
a=rtcp-fb:39 goog-remb
a=rtcp-fb:39 transport-cc
a=rtcp-fb:39 ccm fir
a=rtcp-fb:39 nack
a=rtcp-fb:39 nack pli
a=fmtp:39 level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=4d001f
a=rtpmap:40 rtx/90000
a=fmtp:40 apt=39
a=rtpmap:98 VP9/90000
a=rtcp-fb:98 goog-remb
a=rtcp-fb:98 transport-cc
a=rtcp-fb:98 ccm fir
a=rtcp-fb:98 nack
a=rtcp-fb:98 nack pli
a=fmtp:98 profile-id=0
a=rtpmap:99 rtx/90000
a=fmtp:99 apt=98
a=rtpmap:100 VP9/90000
a=rtcp-fb:100 goog-remb
a=rtcp-fb:100 transport-cc
a=rtcp-fb:100 ccm fir
a=rtcp-fb:100 nack
a=rtcp-fb:100 nack pli
a=fmtp:100 profile-id=2
a=rtpmap:101 rtx/90000
a=fmtp:101 apt=100
a=rtpmap:123 H264/90000
a=rtcp-fb:123 goog-remb
a=rtcp-fb:123 transport-cc
a=rtcp-fb:123 ccm fir
a=rtcp-fb:123 nack
a=rtcp-fb:123 nack pli
a=fmtp:123 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=64001f
a=rtpmap:119 rtx/90000
a=fmtp:119 apt=123
a=rtpmap:114 red/90000
a=rtpmap:115 rtx/90000
a=fmtp:115 apt=114
a=rtpmap:116 ulpfec/90000
a=ssrc-group:FID 2960198605 2305541178
a=ssrc:2960198605 cname:OhkUcqD0lFx1l6U5
a=ssrc:2960198605 msid:5FBEHloS5pG7M3cjTBA8kGFpDcT8gP5CBxif e2932a59-9078-4795-8923-41a8ba7c8b62
a=ssrc:2305541178 cname:OhkUcqD0lFx1l6U5
a=ssrc:2305541178 msid:5FBEHloS5pG7M3cjTBA8kGFpDcT8gP5CBxif e2932a59-9078-4795-8923-41a8ba7c8b62
m=application 9 UDP/DTLS/SCTP webrtc-datachannel
c=IN IP4 0.0.0.0
a=ice-ufrag:RUtG
a=ice-pwd:aMpyBrT2rIfKL8JM+9Q6puft
a=ice-options:trickle
a=fingerprint:sha-256 D7:AC:4D:97:15:C7:17:56:4C:0E:F4:44:CF:0B:EA:E7:32:EF:25:1B:61:83:26:37:DD:52:9F:F5:B9:97:67:FA
a=setup:actpass
a=mid:1
a=sctp-port:5000
a=max-message-size:262144

收到消息：{"type":"candidate","candidate":{"candidate":"candidate:1243366901 1 udp 2113937151 f0898bd7-43e5-4059-8a37-4a56bfd6c23d.local 54214 typ host generation 0 ufrag RUtG network-cost 999","sdpMid":"0","sdpMLineIndex":0},"name":"111","myName":"222"}
Nov 12, 2022 8:23:14 PM io.bison.SignalServer onMessage
INFO: Sending candidate to 111
收到消息：{"type":"candidate","candidate":{"candidate":"candidate:3366687973 1 udp 2113939711 bd535fac-91f4-437c-b974-1cfb716bf3c6.local 54215 typ host generation 0 ufrag RUtG network-cost 999","sdpMid":"0","sdpMLineIndex":0},"name":"111","myName":"222"}
Nov 12, 2022 8:23:14 PM io.bison.SignalServer onMessage
INFO: Sending candidate to 111
收到消息：{"type":"candidate","candidate":{"candidate":"candidate:1243366901 1 udp 2113937151 f0898bd7-43e5-4059-8a37-4a56bfd6c23d.local 54216 typ host generation 0 ufrag RUtG network-cost 999","sdpMid":"1","sdpMLineIndex":1},"name":"111","myName":"222"}
Nov 12, 2022 8:23:14 PM io.bison.SignalServer onMessage
INFO: Sending candidate to 111
收到消息：{"type":"candidate","candidate":{"candidate":"candidate:3366687973 1 udp 2113939711 bd535fac-91f4-437c-b974-1cfb716bf3c6.local 54217 typ host generation 0 ufrag RUtG network-cost 999","sdpMid":"1","sdpMLineIndex":1},"name":"111","myName":"222"}
Nov 12, 2022 8:23:14 PM io.bison.SignalServer onMessage
INFO: Sending candidate to 111
收到消息：{"type":"candidate","candidate":{"candidate":"candidate:2341682919 1 udp 1677729535 223.73.185.58 2090 typ srflx raddr 0.0.0.0 rport 0 generation 0 ufrag RUtG network-cost 999","sdpMid":"0","sdpMLineIndex":0},"name":"111","myName":"222"}
Nov 12, 2022 8:23:14 PM io.bison.SignalServer onMessage
INFO: Sending candidate to 111
收到消息：{"type":"candidate","candidate":{"candidate":"candidate:2341682919 1 udp 1677729535 223.73.185.58 2091 typ srflx raddr 0.0.0.0 rport 0 generation 0 ufrag RUtG network-cost 999","sdpMid":"1","sdpMLineIndex":1},"name":"111","myName":"222"}
Nov 12, 2022 8:23:14 PM io.bison.SignalServer onMessage
INFO: Sending candidate to 111
收到消息：{"type":"answer","answer":{"sdp":"v..."}
Nov 12, 2022 8:23:14 PM io.bison.SignalServer onMessage
INFO: answer from 111

SDP:
v=0
o=- 2339635737454351084 2 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0 1
a=extmap-allow-mixed
a=msid-semantic: WMS 25HJx1gIIdi4rLrJibTArTb91FIXBWrWz5K4
m=video 9 UDP/TLS/RTP/SAVPF 96 97 102 122 127 121 125 107 108 109 124 120 39 40 98 99 100 101 123 119 114 115 116
c=IN IP4 0.0.0.0
a=rtcp:9 IN IP4 0.0.0.0
a=ice-ufrag:UXpl
a=ice-pwd:wM4XWfz5p8j99FhI5CmwKBBU
a=ice-options:trickle
a=fingerprint:sha-256 7A:8E:1F:93:E9:8B:E6:69:A9:28:9D:52:99:C2:44:42:54:F0:7F:12:91:CA:03:17:09:A8:E1:A3:AA:34:34:28
a=setup:active
a=mid:0
a=extmap:1 urn:ietf:params:rtp-hdrext:toffset
a=extmap:2 http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time
a=extmap:3 urn:3gpp:video-orientation
a=extmap:4 http://www.ietf.org/id/draft-holmer-rmcat-transport-wide-cc-extensions-01
a=extmap:5 http://www.webrtc.org/experiments/rtp-hdrext/playout-delay
a=extmap:6 http://www.webrtc.org/experiments/rtp-hdrext/video-content-type
a=extmap:7 http://www.webrtc.org/experiments/rtp-hdrext/video-timing
a=extmap:8 http://www.webrtc.org/experiments/rtp-hdrext/color-space
a=extmap:9 urn:ietf:params:rtp-hdrext:sdes:mid
a=extmap:10 urn:ietf:params:rtp-hdrext:sdes:rtp-stream-id
a=extmap:11 urn:ietf:params:rtp-hdrext:sdes:repaired-rtp-stream-id
a=sendrecv
a=msid:25HJx1gIIdi4rLrJibTArTb91FIXBWrWz5K4 7db0a3c2-656f-4f1a-851b-4792b8172c05
a=rtcp-mux
a=rtcp-rsize
a=rtpmap:96 VP8/90000
a=rtcp-fb:96 goog-remb
a=rtcp-fb:96 transport-cc
a=rtcp-fb:96 ccm fir
a=rtcp-fb:96 nack
a=rtcp-fb:96 nack pli
a=rtpmap:97 rtx/90000
a=fmtp:97 apt=96
a=rtpmap:102 H264/90000
a=rtcp-fb:102 goog-remb
a=rtcp-fb:102 transport-cc
a=rtcp-fb:102 ccm fir
a=rtcp-fb:102 nack
a=rtcp-fb:102 nack pli
a=fmtp:102 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42001f
a=rtpmap:122 rtx/90000
a=fmtp:122 apt=102
a=rtpmap:127 H264/90000
a=rtcp-fb:127 goog-remb
a=rtcp-fb:127 transport-cc
a=rtcp-fb:127 ccm fir
a=rtcp-fb:127 nack
a=rtcp-fb:127 nack pli
a=fmtp:127 level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=42001f
a=rtpmap:121 rtx/90000
a=fmtp:121 apt=127
a=rtpmap:125 H264/90000
a=rtcp-fb:125 goog-remb
a=rtcp-fb:125 transport-cc
a=rtcp-fb:125 ccm fir
a=rtcp-fb:125 nack
a=rtcp-fb:125 nack pli
a=fmtp:125 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=42e01f
a=rtpmap:107 rtx/90000
a=fmtp:107 apt=125
a=rtpmap:108 H264/90000
a=rtcp-fb:108 goog-remb
a=rtcp-fb:108 transport-cc
a=rtcp-fb:108 ccm fir
a=rtcp-fb:108 nack
a=rtcp-fb:108 nack pli
a=fmtp:108 level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=42e01f
a=rtpmap:109 rtx/90000
a=fmtp:109 apt=108
a=rtpmap:124 H264/90000
a=rtcp-fb:124 goog-remb
a=rtcp-fb:124 transport-cc
a=rtcp-fb:124 ccm fir
a=rtcp-fb:124 nack
a=rtcp-fb:124 nack pli
a=fmtp:124 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=4d001f
a=rtpmap:120 rtx/90000
a=fmtp:120 apt=124
a=rtpmap:39 H264/90000
a=rtcp-fb:39 goog-remb
a=rtcp-fb:39 transport-cc
a=rtcp-fb:39 ccm fir
a=rtcp-fb:39 nack
a=rtcp-fb:39 nack pli
a=fmtp:39 level-asymmetry-allowed=1;packetization-mode=0;profile-level-id=4d001f
a=rtpmap:40 rtx/90000
a=fmtp:40 apt=39
a=rtpmap:98 VP9/90000
a=rtcp-fb:98 goog-remb
a=rtcp-fb:98 transport-cc
a=rtcp-fb:98 ccm fir
a=rtcp-fb:98 nack
a=rtcp-fb:98 nack pli
a=fmtp:98 profile-id=0
a=rtpmap:99 rtx/90000
a=fmtp:99 apt=98
a=rtpmap:100 VP9/90000
a=rtcp-fb:100 goog-remb
a=rtcp-fb:100 transport-cc
a=rtcp-fb:100 ccm fir
a=rtcp-fb:100 nack
a=rtcp-fb:100 nack pli
a=fmtp:100 profile-id=2
a=rtpmap:101 rtx/90000
a=fmtp:101 apt=100
a=rtpmap:123 H264/90000
a=rtcp-fb:123 goog-remb
a=rtcp-fb:123 transport-cc
a=rtcp-fb:123 ccm fir
a=rtcp-fb:123 nack
a=rtcp-fb:123 nack pli
a=fmtp:123 level-asymmetry-allowed=1;packetization-mode=1;profile-level-id=64001f
a=rtpmap:119 rtx/90000
a=fmtp:119 apt=123
a=rtpmap:114 red/90000
a=rtpmap:115 rtx/90000
a=fmtp:115 apt=114
a=rtpmap:116 ulpfec/90000
a=ssrc-group:FID 3498837450 3401073440
a=ssrc:3498837450 cname:1TsjTTcNusiwh1dy
a=ssrc:3401073440 cname:1TsjTTcNusiwh1dy
m=application 9 UDP/DTLS/SCTP webrtc-datachannel
c=IN IP4 0.0.0.0
a=ice-ufrag:UXpl
a=ice-pwd:wM4XWfz5p8j99FhI5CmwKBBU
a=ice-options:trickle
a=fingerprint:sha-256 7A:8E:1F:93:E9:8B:E6:69:A9:28:9D:52:99:C2:44:42:54:F0:7F:12:91:CA:03:17:09:A8:E1:A3:AA:34:34:28
a=setup:active
a=mid:1
a=sctp-port:5000
a=max-message-size:262144

Nov 12, 2022 8:23:14 PM io.bison.SignalServer onMessage
INFO: send answer to 222

收到消息：{"type":"candidate","candidate":{"candidate":"candidate:3031372602 1 udp 2113937151 d05702cc-9023-4171-89d2-d6c7a23e72b5.local 49996 typ host generation 0 ufrag UXpl network-cost 999","sdpMid":"0","sdpMLineIndex":0},"name":"222","myName":"111"}
Nov 12, 2022 8:23:14 PM io.bison.SignalServer onMessage
INFO: Sending candidate to 222
收到消息：{"type":"candidate","candidate":{"candidate":"candidate:4046756185 1 udp 2113939711 db8608fa-054c-4468-b47b-5fe29f325086.local 49997 typ host generation 0 ufrag UXpl network-cost 999","sdpMid":"0","sdpMLineIndex":0},"name":"222","myName":"111"}
Nov 12, 2022 8:23:14 PM io.bison.SignalServer onMessage
INFO: Sending candidate to 222
收到消息：{"type":"candidate","candidate":{"candidate":"candidate:3337783342 1 udp 1677729535 223.73.185.58 2092 typ srflx raddr 0.0.0.0 rport 0 generation 0 ufrag UXpl network-cost 999","sdpMid":"0","sdpMLineIndex":0},"name":"222","myName":"111"}
Nov 12, 2022 8:23:14 PM io.bison.SignalServer onMessage
INFO: Sending candidate to 222
```

可以看到，candidate中含有223.73.185.58这个地址，是我两个peer电脑的共同的移动宽带出口。如果关闭stun服务，是找不到这样一个IP的，candidate里只有局域网内的IP信息，但视频也能通，因为在同一个局域网内。



### 5、 参考资料

```
https://blog.csdn.net/lym594887256/article/details/124081420
https://blog.csdn.net/wangxudongx/article/details/105414629
https://blog.csdn.net/wangxudongx/article/details/105447416
https://javascript.info/websocket
https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API/Simple_RTCDataChannel_sample
```

还缺一点功夫，待续：

1. 想用自己的熟悉的语言来开发peer端，不想用js，不太会。看起来安卓是有库支持的。
2. 怎么实现cs视频通信，也就是server也作为一个peer端参与进来？