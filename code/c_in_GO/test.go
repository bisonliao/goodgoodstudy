// test
package main

import (
	"fmt"
	"math/rand"
	"net"
	"os"
	"strconv"
	"time"
	"unsafe"
)
// import c要和上面的注释紧紧挨着，不要有空行

/*
#include "business.h"

int do_business(const unsigned char* req,  int reqlen,
		unsigned char * resp,  int max_resp_len); 
*/
import "C"  

type Msg struct {
	data    []byte
	datalen int
}

/*
提供给c语言写的业务代码用来对外通信的两个函数：发请求包然后收应答包
int MSG_BUS_SendRequest(const char * svc_name, int svc_name_len,
	   				const char * intf_name, int intf_name_len,
	   				const unsigned char * request,
	    				int req_len);
int MSG_BUS_RecvResponse(unsigned char * response, int resp_max_len);
*/
//export MSG_BUS_SendRequest
func MSG_BUS_SendRequest(svc_name unsafe.Pointer, svc_name_len C.int,
	inf_name unsafe.Pointer, inf_name_len C.int,
	request unsafe.Pointer, req_len C.int) int {
	
	fmt.Println("in MSG_BUS_SendRequest");

	return 0
}
//export  MSG_BUS_RecvResponse
func MSG_BUS_RecvResponse(response unsafe.Pointer, resp_max_len C.int) int {
	fmt.Println("in MSG_BUS_RecvResponse");
	return 0
}

//调用c语言写的业务代码
func Do_Business(request []byte) ([]byte, int) {

	var req_point *C.uchar = (*C.uchar)(&request[0])
	var reqlen C.int = (C.int)(len(request))

	var resp_byte []byte = make([]byte, 1024)
	var resp_point *C.uchar = (*C.uchar)(&resp_byte[0])
	

	var resp_len C.int = C.do_business(req_point, reqlen,
		resp_point, 1024)

	return resp_byte, int(resp_len)

}

func work(c1 chan Msg, c2 chan Msg, work_id int) {
	var tmpch chan Msg
	tmpch = make(chan Msg, 100)
	for {
		msg := <-c1
		var s string

		s = string(msg.data)
		t1 := time.Now()
		//fmt.Println("!!read ", s, " by ", work_id)
		//超时100ms，模拟业务处理过程远程访问的耗时
		select {
		case <-time.After(time.Millisecond * 100):
		case <-tmpch:
		}

		Do_Business(msg.data);

		//应答客户端
		t2 := time.Now()
		diff := t2.Sub(t1)

		var dur int64 = diff.Nanoseconds() / 1000000
		s = s + " dur:" + string(dur)
		//抽样输出处理耗时
		if rand.Intn(500000) == 1 {
			fmt.Println(dur)
		}

		var resp_msg Msg
		resp_msg.data = []byte(s)
		resp_msg.datalen = len(resp_msg.data)

		c2 <- resp_msg

	}
}

func main() {
	const ip = "0.0.0.0"
	port := 44000
	const work_num = 30000
	//获得命令行参数中的端口
	arg_num := len(os.Args)
	fmt.Println("argnum:", arg_num)
	fmt.Print("arg[1]:", os.Args[1])
	if arg_num == 2 {
		var err error
		port, err = strconv.Atoi(os.Args[1])
		if err == nil {
			fmt.Println("use port:", port)
		}
	}

	//监听udp
	addr, err := net.ResolveUDPAddr("udp", ip+":"+strconv.Itoa(port))
	if err != nil {
		fmt.Println("net.ResolveUDPAddr fail.", err)
		return
	}
	conn, err := net.ListenUDP("udp", addr)
	if err != nil {
		fmt.Println("net.ListenUDP fail.", err)
		return
	}
	defer conn.Close()

	//创建1w个工人
	var resp_channel chan Msg
	resp_channel = make(chan Msg, 100)
	var i int
	var req_channel [work_num]chan Msg
	for i = 0; i < work_num; i++ {
		req_channel[i] = make(chan Msg, 100)

		go work(req_channel[i], resp_channel, i+1)
	}
	var cnt uint32 = 0
	var prev_min int = time.Now().Minute()

	for {
		//带超时时间的收请求包
		buf := make([]byte, 1024)
		deadline := time.Now().Add(1 * time.Millisecond)
		conn.SetReadDeadline(deadline)
		len, addr, err := conn.ReadFromUDP(buf)

		if err == nil && len > 5 {
			var msg Msg
			msg.data = buf[0:len]
			msg.datalen = len

			var index int
			index = rand.Intn(work_num)
			req_channel[index] <- msg
		}

		//带超时时间的查看应答包

		select {
		case <-time.After(time.Millisecond * 1):

		case resp := <-resp_channel:
			{
				conn.WriteToUDP(resp.data, addr)

				//打印统计信息
				cnt++
				cur_min := time.Now().Minute()
				if cur_min != prev_min {
					fmt.Println("in one minute, server proceeds ", cnt)
					cnt = 0
					prev_min = cur_min
				}

			}
		}

	}
}
