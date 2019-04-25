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

(*random init w1=[w1,w2] and b1=[b]*)
initParam[] := Module[{},
   w1 = Array[RandomReal[{-1.414, 1.414}] &, {1, 2}];
   b1 = Array[0 &, {1, 1}];
   ];
(* random init some points and their lables *)
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

forward[] := Module[{i},

   z1 = w1.a0 + b1.{Table[1, {i, 1, m}]};
   a1 = sigmoid[z1];
   ];

cost[a_, y_] := Module[{l},
   l = -(y*Log[ a ] + (1 - y)*Log[1 - a]);

   l = Total[l]/Length[a];

   Return[l];
   ];

main[] := Module[{i, j, c},

   fetchBatch[];
   points = {};
   Do[
    initParam[];
    forward[];
         c = cost[a1[[1]], y[[1]]];
    points = Join[points, {{w1[[1]][[1]], w1[[1]][[2]], c}}];
    ,
    {i, 1, 10000}
    ];
   Print[points[[1 ;; 4]]];
   Print[ListPlot3D[points]];
   ];
main[];
```