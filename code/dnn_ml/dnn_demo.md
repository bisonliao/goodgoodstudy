下面是一段深度神经网络的demo代码，有两个隐含层，节点数分别是4、3。

输入层节点数是2，输出层节点数是1。训练数据是一段两个同心圆环，外侧的环上的落点分类为1，内侧的环上的落点为0（见fetchBatch函数）。


```
ClearAll["Global`*"];
(*hyperparameters*)
m = 1000;
lr = 0.008;
(*input,layer 0*)
a0 = {};
y = {}
(*layer 1*)
z1 = {};
a1 = {};
w1 = {};
b1 = {};
(*layer 2*)
z2 = {};
a2 = {};
w2 = {};
b2 = {};
(*layer 3*)
z3 = {};
a3 = {};
w3 = {};
b3 = {};
(* init parameters*)
init[] := Module[{},
   w1 = Array[RandomReal[] &, {4, 2}];
   b1 = Array[0 &, {4, 1}];
   w2 = Array[RandomReal[] &, {3, 4}];
   b2 = Array[0 &, {3, 1}];
   w3 = Array[RandomReal[] &, {1, 3}];
   b3 = Array[0 &, {1, 1}];

   ];
(*fetch a batch of train data *)
fetchBatch[] := Module[{r, a, x1, x2, label, i},
   a0 = {};
   y = {};
   Do[
    
    r = RandomReal[1];
    a = RandomReal[2*Pi];
    x1 = r*Cos[a];
    x2 = r*Sin[a];
    If[r > 0.5, label = 1, label = 0];
    AppendTo[a0, {x1, x2}];
    AppendTo[y, {label}];
    ,
    {i, 1, m}
    ];
   a0 = Transpose[a0];
   y = Transpose[y];

   ];

sigmoid[z_] := Module[{ret},
   ret = Table[1/(1 + Exp[-z[[i]]]), {i, 1, Length[z]}];
   Return[ret];
   ];
(*derivative of sigmoid *)
dsigmoid[z_] := Module[{f},
   f = sigmoid[z];
   f*(Array[1 &, Dimensions[z]] - f)
   ];

forward[] := Module[{i},
   (*bi is broadcoasting by multiplied {1,1,...,1} *)
   z1 = w1.a0 + b1.{Table[1, {i, 1, m}]};
   a1 = sigmoid[z1];


   z2 = w2.a1 + b2.{Table[1, {i, 1, m}]};
   a2 = sigmoid[z2];


   z3 = w3.a2 + b3.{Table[1, {i, 1, m}]};
   a3 = sigmoid[z3];

   ];
(* if layer#3 shape changes, cost[] and dcost[] should be modified*)
cost[a_, y_] := Module[{l, ret},

   l = -(y*Log[ a ] + (1 - y)*Log[1 - a]);
   (* sum of row#1, and average *)
   ret = Total[l[[1]]]/Length[   l[[1]]   ];
   Return[ret];
   ];
(*diverative of cost *)
dcost[a_, y_] := Module[{l},
   l = (1 - y)/(1 - a) - y/a;
   Return[l];
   ];
backward[] := Module[{dz3, dw3, db3, dz2, dw2, db2, dz1, dw1, db1, i},
   dz3 = a3 - y;
   dw3 = dz3 . Transpose[a2]/m;
   db3 = Table[{Total[dz3[[i]]]}, {i, 1, Length[dz3]}]/m;

   dz2 = Transpose[dw3].dz3*dsigmoid[z2];
   dw2 = dz2 . Transpose[a1]/m;
   db2 = Table[{Total[dz2[[i]]]}, {i, 1, Length[dz2]}]/m;

   dz1 = Transpose[dw2].dz2*dsigmoid[z1];
   dw1 = dz1 . Transpose[a0]/m;
   db1 = Table[{Total[dz1[[i]]]}, {i, 1, Length[dz1]}]/m;

   w3 = w3 - lr * dw3;
   b3 = b3 - lr * db3;

   w2 = w2 - lr * dw2;
   b2 = b2 - lr * db2;

   w1 = w1 - lr * dw1;
   b1 = b1 - lr * db1;
   ];
main[] := Module[{},
   init[];
   Do[
    If[Mod[i, 1000] == 1, fetchBatch[], null];
    forward[];
    
    If [ Mod[i, 10000] == 1, c = cost[a3, y]; Print[c]; 
     If[c < 0.01, Break[], null];, null];
    backward[];
    ,
    {i, 1 , 1000000}
    ];
   Print["cost:", c];
   Print[MatrixForm[a3]];
   Print[MatrixForm[y]];
   ];
(* 
  check if the train data can be trained as a logistic regression \
model  successfully .
yes, it can .
*)
checkModel[] := Module[{},
   fetchBatch[];
   a0 = Transpose[a0];
   y = Transpose[y];
   td = Table[a0[[i]] -> y[[i]][[1]], {i, 1, m}];
   Print[td[[1 ;; 5]]];
   c = Classify[td];
   Print[c[{0.1, 0.5}]];
   Print[c[{0.1, 0.2}]];
   Print[c[{-0.6, 0.5}]];

   ];

main[];
```