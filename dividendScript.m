%Expected Quarterly earnings: 
earningsPerShare=[
0.54;
0.42;
0.15;
0.93;
0.69;
0.6;
0.15;
0.33;
0.46];

pricePerShare=[
19.17;
12.38;
5.35;
15.94;
23.7;
11.13;
6.62;
17.11;
18.32];

totalInvested=300;

[X,FVAL,EXITFLAG,OUTPUT] = linprog(earningsPerShare,pricePerShare',totalInvested)