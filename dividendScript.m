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

earningsPerDollar = earningsPerShare./pricePerShare;
[~,stockInd] = max(earningsPerDollar);
totalInvested=500;
moneySpent = 0;
moneyLeft = totalInvested-moneySpent;

[vals,indsInOrder] = sort(earningsPerDollar,'descend');
numStocks = zeros(size(earningsPerDollar));
for i = 1:length(numStocksPossible)
   curNumStocks = floor(moneyLeft/pricePerShare(indsInOrder(i)));
   moneySpent = moneySpent + curNumStocks*pricePerShare(indsInOrder(i));
   numStocks(indsInOrder(i))=numStocks(indsInOrder(i))+curNumStocks;
   moneyLeft = totalInvested-moneySpent;
end

totalEarnings = sum(numStocks.*earningsPerShare)
totalCost = sum(numStocks.*pricePerShare)
%{
numStocksPossible = floor(300./pricePerShare);

earnings = earningsPerShare(stockInd)*numStocksPossible(stockInd)
moneySpent = pricePerShare(stockInd)*numStocksPossible(stockInd)



moneySpent = 0;
curValInd = 1;

while(moneySpent < totalInvested)
    curStockInd = indsInOrder(curValInd);
    moneyLeft = totalInvested-moneySpent;
    while(moneyLeft > pricePerShare(curStockInd))
        moneySpent = moneySpent + pricePerShare(curStockInd);
        moneyLeft = totalInvested-moneySpent;
        numStocks(curStockInd)=numStocks(curStockInd)+1;
    end
    curStockInd = curStockInd+1;
end
%}