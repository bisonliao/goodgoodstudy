下面这段代码显示逻辑斯蒂回归的cost函数是w的凸函数：
```
ClearAll["Global`*"];
(*input,layer 0*)

m = 1000;
lr = 1;
a0 = {};
y = {};

(*layer 1*)

z1 = {};
a1 = {};
w1 = {};
b1 = {};
dw1 = {};
db1 = {};

initParam[] := Module[{},
   
   w1 = Array[RandomReal[{-1.414, 1.414}] &, {1, 2}];
   b1 = Array[0 &, {1, 1}];
   ];
fetchBatch[] := Module[{r, a, x1, x2, label, i},
   a0 = {};
   y = {};
   Do[
    
    (*同心圆环上的点的分类，lr对其不能收敛*)
    
    r = RandomReal[1];
    a = RandomReal[2*Pi];
    x1 = r*Cos[a];
    x2 = r*Sin[a];
    If[r > 0.5, label = 1, label = 0];
    
    
    (*最简单的线性分类*)
    (*
    x1 = RandomReal[{-2,2}];
    x2 = RandomReal[{-2,2}];
    If[x1>x2, label=1,label=0];
    *)
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

forward[] := Module[{i},
   
   
   z1 = w1.a0 + b1.{Table[1, {i, 1, m}]};
   
   a1 = sigmoid[z1];
   
   
   
   
   
   ];

cost[a_, y_] := Module[{l},
   l = -(y*Log[ a ] + (1 - y)*Log[1 - a]);
   
   l = Total[l]/Length[a];
   
   Return[l];
   ];
dcost[a_, y_] := Module[{l, ret},
   l = (1 - y)/(1 - a) - y/a;
   ret = l;
   
   
   
   Return[ret];
   ];

dsigmoid[z_] := Module[{f},
   f = sigmoid[z];
   f*(Array[1 &, Dimensions[z]] - f)
   ];
backward[d_] := Module[{dz1, i, dsig},
   
   da1 = dcost[a1[[1]], y[[1]]];
   
   If[d > 0, Print[]; Print["da1:", da1[[1 ;; 6]]], null];
   (*
   dsig =  a1[[1]]*(Table[1,{i,1,m}]-a1[[1]]) ;
   *)
   dsig = dsigmoid[z1][[1]];
   If[d > 0, Print["dsig:", dsig[[1 ;; 6]]], null];
   dz1 = da1 * dsig   ;
   
   If[d > 0, Print["dz1:", dz1], null];
   If[d > 0, Print["a0:", a0], null];
   
   
   dw1 = {dz1} . Transpose[a0]/m;
   If[d > 0, Print["dw1:", dw1, " ", m], null];
   
   db1 = Total[dz1]/m;
   
   
   w1 = w1 - lr * dw1;
   b1 = b1 - lr * db1;
   
   ];
(* train the model *)

train[] := Module[{i, j, c, debug},
   
   initParam[];
   
   
   debug = 0;
   Do[
    forward[];
    If[Mod[i, 100] == 7, c = cost[a1[[1]], y[[1]]]; 
     Print["cost:", c, dw1, w1]; debug = 0, debug = 0];
    
    backward[debug];
    ,
    {i, 1, 1000}
    ];
   c = cost[a1[[1]], y[[1]]];
   
   Print["cost:", c];
   
   
   Print[MatrixForm[w1], MatrixForm[b1]];
   ];

(* test the accuracy *)

test[] := Module[{ycap},
   fetchBatch[];
   forward[];
   
   ycap = Table[Floor[a1[[1]][[i]] + 0.5], {i, 1, m}];
   N[Total[Table[If[y[[1, i]] == ycap[[i]], 1, 0], {i, 1, m}]]/m]
   
   ];
(* 用成熟的训练方法交叉验证可行性和正确性 *)
checkModel[] := Module[{testset, trainset, c},
   
   fetchBatch[];
   testset = Table[a0[[;; , i]] -> y[[1, i]], {i, 1, m}];
   fetchBatch[];
   trainset = Table[a0[[;; , i]] -> y[[1, i]], {i, 1, m}];
   
   c = Classify[trainset, Method -> "LogisticRegression"];
   Print[c];
   Print["accuracy:", ClassifierMeasurements[c, testset, "Accuracy"]];
   ];
(* show the cost function is convex of parameter w *)
showConvex[] := Module[{i, points, c, dc},
   
   
   points = {};
   points2 = {};
   Do[
    initParam[];
    
    forward[];
    c = cost[a1[[1]], y[[1]]];
    
    backward[0];
    points = Join[points, {{w1[[1]][[1]], w1[[1]][[2]], c}}];
    points2 = 
     Join[points2, {{w1[[1]][[1]], w1[[1]][[2]], dw1[[1]][[1]]}}];
    ,
    {i, 1, 10000}
    ];
   
   Print[ListPlot3D[points]];
   Print[ListPlot3D[points2]];
   
   ];

(* show the cost function is convex of parameter w *)
showConvex2[] := Module[{i, points, c, dc, w1, w2, res},
   
   
   points = {};
   points2 = {};
   fz[x1_, x2_, w1_, w2_] := x1 * w1 + x2 * w2;
   fg[z_] := 1/(1 + Exp[-z]);
   fl[a_, y_] := -(y*Log[a] + (1 - y)*Log[1 - a]);
   
   fc[w1_, w2_] := 
    Sum[ fl[fg[   fz[a0[[1, i]], a0[[2, i]], w1, w2]   ], 
       y[[1, i]]], {i, 1, m}]/m;
   
   
   fd[w1_, w2_] = 
    D[Sum[ fl[fg[   fz[a0[[1, i]], a0[[2, i]], w1, w2]   ], 
        y[[1, i]]], {i, 1, m}]/m, w1];
   
   
   Do[
    
    w1 = RandomReal[{-1.414, 1.414}];
    w2 = RandomReal[{-1.414, 1.414}];
    res = fc[w1, w2];
    AppendTo[points, {w1, w2, res}];
    res = fd[w1, w2];
    AppendTo[points2, {w1, w2, res}];
    ,
    {i, 1, 10000}
    ];
   
   Print[ListPlot3D[points]];
   Print[ListPlot3D[points2]];
   
   
   ];

fetchBatch[];
showConvex[];
train[];
test[]
```