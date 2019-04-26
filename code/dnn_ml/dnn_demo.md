下面是一段深度神经网络的demo代码，有两个隐含层，节点数分别是4、3。

输入层节点数是2，输出层节点数是1。训练数据是两个同心圆环上的点的坐标和分类，外侧的环上的落点分类为1，内侧的环上的落点为0（见fetchBatch函数）。


```
ClearAll["Global`*"];
(*hyperparameters*)
m = 10000;
lr = 0.008;
(*input,layer 0*)
a0 = {};
y = {};
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
(*init parameters*)
init[] := Module[{},
   w1 = Array[RandomReal[{-Sqrt[6/6], Sqrt[6/6] }] &, {4, 2}];
   b1 = Array[0 &, {4, 1}];
   w2 = Array[RandomReal[{-Sqrt[6/7], Sqrt[6/7] }] &, {3, 4}];
   b2 = Array[0 &, {3, 1}];
   w3 = Array[RandomReal[{-Sqrt[6/4], Sqrt[6/4] }] &, {1, 3}];
   b3 = Array[0 &, {1, 1}];];
(*fetch a batch of train data*)
fetchBatch[] := Module[{r, a, x1, x2, label, i},
   
   a0 = {};
   y = {};
   Do[
    (*
    r=RandomReal[1];
    a=RandomReal[2*Pi];
    x1=r*Cos[a];
    x2=r*Sin[a];
    If[r>0.5,label=1,label=0];
    *)
    x1 = RandomReal[{-5, 5}];
    x2 = RandomReal[{-5, 5}];
    If[x1 > x2, label = 1, label = 0];
    AppendTo[a0, {x1, x2}];
    AppendTo[y, {label}];, {i, 1, m}];
   a0 = Transpose[a0];
   y = Transpose[y];];

sigmoid[z_] := Module[{ret},
   ret = Table[   1/(1 + Exp[-z[[i]]]), {i, 1, Length[z]}];
   Return[ret];];
(*derivative of sigmoid*)
dsigmoid[z_] := Module[{f},
   f = sigmoid[z];
   f*(Array[1 &, Dimensions[z]] - f)
   ];
relu[z_] := Module[{ret},
   ret = Array[Max[z[[#1, #2]], 0.001*z[[#1, #2]]] &, Dimensions[z]];
   Return[ret];
   ];
drelu[z_] := Module[{ret},
   ret = Array[If[z[[#1, #2]] >= 0, 1, 0.001] &, Dimensions[z]];
   Return[ret];
   ];
   
forward[] := Module[{i},
   (*bi is broadcoasting by multiplied {1,1,...,1}*)
   
   z1 = w1.a0 + b1.{Table[1, {i, 1, m}]};
   a1 = sigmoid[z1];
   z2 = w2.a1 + b2.{Table[1, {i, 1, m}]};
   a2 = sigmoid[z2];
   z3 = w3.a2 + b3.{Table[1, {i, 1, m}]};
   a3 = sigmoid[z3];
   ];
(*if layer#3 shape changes,cost[] and dcost[] should be modified*)
cost[a_, y_] := Module[{l, ret},
   l = -(y*Log[a] + (1 - y)*Log[1 - a]);
   (*sum of row#1,and average*)
   ret = Total[l[[1]]]/Length[l[[1]]];
   Return[ret];];
(*diverative of cost*)
dcost[a_, y_] := Module[{l},
   l = (1 - y)/(1 - a) - y/a;
   Return[l];
   ];
flag = 1;
backward[] := 
  Module[{dz3, dw3, db3, dz2, dw2, db2, dz1, dw1, db1, da3, i},
   da3 = dcost[a3, y];
   If[flag > 0, Print["da3:", Dimensions[da3]], null];
   dz3 = da3*dsigmoid[z3];
   If[flag > 0, Print["dz3:", Dimensions[dz3]], null];
   dw3 = dz3.Transpose[a2]/m;
   If[flag > 0, Print["dw3:", Dimensions[dw3]], null];
   
   db3 = Table[{Total[dz3[[i]]]}, {i, 1, Length[dz3]}]/m;
   If[flag > 0, Print["db3:", Dimensions[db3]], null];
   dz2 = Transpose[dw3].dz3*dsigmoid[z2];
   If[flag > 0, Print["dz2:", Dimensions[dz2]], null];
   dw2 = dz2.Transpose[a1]/m;
   If[flag > 0, Print["dw2:", Dimensions[dw2]], null];
   db2 = Table[{Total[dz2[[i]]]}, {i, 1, Length[dz2]}]/m;
   If[flag > 0, Print["db2:", Dimensions[db2]], null];
   dz1 = Transpose[dw2].dz2*dsigmoid[z1];
   If[flag > 0, Print["dz1:", Dimensions[dz1]], null];
   dw1 = dz1.Transpose[a0]/m;
   If[flag > 0, Print["dw1:", Dimensions[dw1]], null];
   db1 = Table[{Total[dz1[[i]]]}, {i, 1, Length[dz1]}]/m;
   If[flag > 0, Print["db1:", Dimensions[db1]], null];
   w3 = w3 - lr*dw3;
   b3 = b3 - lr*db3;
   w2 = w2 - lr*dw2;
   b2 = b2 - lr*db2;
   w1 = w1 - lr*dw1;
   b1 = b1 - lr*db1;
   
   flag = 0;
   ];
train[] := Module[{},
   init[];
   fetchBatch[];
   Do[
    
    forward[];
    If[Mod[i, 100] == 1, c = cost[a3, y]; Print[c];
     If[c < 0.01, Break[], null];, null];
    backward[];,
    {i, 1, 10000}
    ];
   Print["cost:", c];
   
   ];
test[] := Module[{ycap,acc},
   fetchBatch[];
   forward[];
   
   ycap = Table[Floor[a3[[1]][[i]] + 0.5], {i, 1, m}];
   acc = N[Total[Table[If[y[[1, i]] == ycap[[i]], 1, 0], {i, 1, m}]]/m];
   Print["accuracy:", acc];
   
   ];
(*check if the train data can be trained as a logistic regression \
model successfully.yes,it can.*)
checkModel[] := Module[{}, fetchBatch[];
   fetchBatch[];
   testset = Table[a0[[;; , i]] -> y[[1, i]], {i, 1, m}];
   fetchBatch[];
   trainset = Table[a0[[;; , i]] -> y[[1, i]], {i, 1, m}];
   
   c = Classify[trainset, Method -> "NeuralNetwork"];
   Print[c];
   Print["accuracy:", ClassifierMeasurements[c, testset, "Accuracy"]];
   ];

train[];
test[]
```



对于简单的线性分类（x1 > x2时分类为1，否则分类为0），经过一万次迭代，每次迭代batch size为10000，模型的准确率可以到86.8%：

```
0.793637
da3:{1,10000}
dz3:{1,10000}
dw3:{1,3}
db3:{1,1}
dz2:{3,10000}
dw2:{3,4}
db2:{3,1}
dz1:{4,10000}
dw1:{4,2}
db1:{4,1}
0.738187
0.710133
0.696196
0.689112
...
0.611565
0.611006
0.61045
cost:0.61045
accuracy:0.8686
```

对于同心圆环上的这种复杂点的非线性分类，模型也能到70%的准确率：

```
0.751562
da3:{1,1000}
dz3:{1,1000}
dw3:{1,3}
db3:{1,1}
dz2:{3,1000}
dw2:{3,4}
db2:{3,1}
dz1:{4,1000}
dw1:{4,2}
db1:{4,1}
0.738357
0.72885
0.721566
...
0.597903
0.597756
0.597619
0.597486
cost:0.597486
accuracy:0.706
```

