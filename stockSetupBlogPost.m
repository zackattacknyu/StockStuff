%{
This is the S&P500 data from the data set
put into a .mat file. 
%}
load('SAP500DataForOneYear.mat');

%Feature data will be extracted from Jan 10 - Jan 29 stock info
%Target will be the fractional change in 
%	closing price from Jan 29 2010 to Feb 10 2010
trainingInds = find(Date>20100110 & Date<20100200);
targetInds = find(Date==20100129 | Date==20100210);

symbols = unique(Symbol);
trainData = cell(1,length(symbols));
trainDataTarget = zeros(1,length(symbols));
XdataOther = zeros(length(symbols),11);
goodInds = [];

%go through each stock and assemble the features and targets
for ii = 1:length(symbols)
    curSymbol = symbols(ii);
    symbolInds = find(strcmp(Symbol,curSymbol));
    curTrainInds = intersect(trainingInds,symbolInds);
    curTargetInds = intersect(targetInds,symbolInds);
    if(length(curTrainInds)<1 || length(curTargetInds) < 2)
       continue 
    end
    goodInds = [goodInds ii];
    targetData = Close(curTargetInds);
	
	%obtains the fractional change in price for the stock
    currentY = (targetData(2)-targetData(1))/targetData(1); 
	trainDataTarget(ii) = currentY;
    curTrain = zeros(length(curTrainInds),5);
	
	%{
	Due to varying prices between stocks, all feature data that is a price
		is a fraction of the opening price of the stock. 
	%}
    openPrice = Open(curTrainInds(1)); 
	
	%{
	Old Feature Data Regime:
	Training Data is all the opening, closing, high, and low prices for the stock 
	%}
    curTrain(:,1)=Open(curTrainInds);
    curTrain(:,2)=Close(curTrainInds);
    curTrain(:,3)=Low(curTrainInds);
    curTrain(:,4)=High(curTrainInds);
    curTrain(:,1:4)=(curTrain(:,1:4)-openPrice)/openPrice;
    curTrain(:,5)=Volume(curTrainInds);
    trainData{ii} = curTrain;
    
	%{
	My current feature data regime: 
		min, max, mean, and std of opening price, closing price, and volume on each day
		min of the lowest price for the stock
		max of the highest price for the stock
	%}
    XdataOther(ii,1)=min(curTrain(:,1));
    XdataOther(ii,2)=max(curTrain(:,1));
    XdataOther(ii,3)=mean(curTrain(:,1));
    XdataOther(ii,4)=std(curTrain(:,1));
    
    XdataOther(ii,5)=min(curTrain(:,2));
    XdataOther(ii,6)=max(curTrain(:,2));
    XdataOther(ii,7)=mean(curTrain(:,2));
    XdataOther(ii,8)=std(curTrain(:,2));
    
    XdataOther(ii,9)=min(curTrain(:,5));
    XdataOther(ii,10)=max(curTrain(:,5));
    XdataOther(ii,11)=mean(curTrain(:,5));
    XdataOther(ii,12)=std(curTrain(:,5));
    
    XdataOther(ii,13)=min(curTrain(:,3));
    XdataOther(ii,14)=max(curTrain(:,4));
end

XdataOther = XdataOther(goodInds,:);
trainDataAdj = trainData(goodInds);
Xdata = zeros(length(trainDataAdj),numel(trainDataAdj{1}));
for i = 1:length(trainDataAdj)
    curFeat = trainDataAdj{i};
    Xdata(i,:) = curFeat(:)';
end
Ydata = trainDataTarget(goodInds);
symbolsUsed = Symbol(goodInds);

%split into training and validation set
randInds = randperm(length(symbolsUsed));
numTrain = floor(length(symbolsUsed)*0.8);
trainInds = randInds(1:numTrain);
testInds = randInds(numTrain+1:end);

Xtrain = XdataOther(trainInds,:);
Xtest = XdataOther(testInds,:);
Ytrain = Ydata(trainInds);
Ytest = Ydata(testInds);

%{
This saves the target and feature data to a MAT file
	that the Python script will use for the data
	to put into XGBoost
%}
%save('TrainTestData4.mat','Xtrain','Xtest','Ytrain','Ytest');

%{
This loads the result of the Python script
%}
%load('stockResults3.mat');

%{
XGBoost outputs the RMSE of our algorithm. 
As a baseline, this calculates the RMSE if 
	random numbers were chosen. 
When I ran this, it produced an RMSE of 0.0696 if random
	numbers were chosen based on the mean and std
	of training set. 
XGBoost gave me an RMSE of 0.038
%}
YdataRand = randn(size(Ytrain)).*std(Ytrain)+mean(Ytrain);
sqrt(sum((Ytrain-YdataRand).^2)/length(Ytrain))


%{
These graphs show the spread of target/prediction fractions
%}
[~,II] = sort(Ytest);
[~,II2] = sort(yHatTest);

figure
hold on
plot(Ytest(II),'k-')
plot(yHatTest(II),'ro')
plot(zeros(1,100),'g--')
legend('Actual Stock Price Change','Predicted Stock Price Change');
hold off

figure
hold on
plot(Ytest(II2),'ko')
plot(yHatTest(II2),'r-')
plot(zeros(1,100),'g--')
legend('Actual Stock Price Change','Predicted Stock Price Change');
hold off

%{
Now I test the trading strategy
%}

%if the prediction is above this value, buy 1 share of the stock
thresholdToBuy = 0.04 

%if the prediction is below this value, short 1 share of the stock
thresholdToShort = -0.05

ii = find(yHatTest>thresholdToBuy);
originalInds = goodInds(testInds(ii));

ii2 = find(yHatTest<thresholdToShort);
origInds2 = goodInds(testInds(ii2));

moneyMade = 0;
moneySpent = 0;

%money we made by buying stock
for jj = originalInds
    curSymbol = symbols(jj);
    symbolInds = find(strcmp(Symbol,curSymbol));
    curTargetInds = intersect(targetInds,symbolInds);
    if(length(curTrainInds)<1 || length(curTargetInds) < 2)
       continue 
    end
    targetData = Close(curTargetInds);
    moneyMade = moneyMade + (targetData(2)-targetData(1));
    moneySpent = moneySpent + targetData(1);
end

%money we make by shorting stock
for jj = origInds2
    curSymbol = symbols(jj);
    symbolInds = find(strcmp(Symbol,curSymbol));
    curTargetInds = intersect(targetInds,symbolInds);
    if(length(curTrainInds)<1 || length(curTargetInds) < 2)
       continue 
    end
    targetData = Close(curTargetInds);
    moneyMade = moneyMade + (targetData(1)-targetData(2));
    moneySpent = moneySpent + targetData(2);
end

%print out money made versus money spent
moneyMade
moneySpent

%print the fractional return on investment
ROI=moneyMade/moneySpent