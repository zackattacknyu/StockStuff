load('SAP500DataForOneYear.mat');

%training set will be Jan 10 - Jan 29 data

%target will be predicting whether closing price on February 10 2010
%   was higher or lower than closing price on January 29 2010

trainingInds = find(Date>20100110 & Date<20100200);
targetInds = find(Date==20100129 | Date==20100210);

symbols = unique(Symbol);
trainData = cell(1,length(symbols));
trainDataTarget = zeros(1,length(symbols));
XdataOther = zeros(length(symbols),11);
goodInds = [];
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
    currentY = (targetData(2)-targetData(1))/targetData(1);
    trainDataTarget(ii) = currentY;
    curTrain = zeros(length(curTrainInds),5);
    openPrice = Open(curTrainInds(1));
    curTrain(:,1)=Open(curTrainInds);
    curTrain(:,2)=Close(curTrainInds);
    curTrain(:,3)=Low(curTrainInds);
    curTrain(:,4)=High(curTrainInds);
    
    curTrain(:,1:4)=(curTrain(:,1:4)-openPrice)/openPrice;
    
    curTrain(:,5)=Volume(curTrainInds);
    trainData{ii} = curTrain;
    
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
    
    %NOTE: INCLUDE THE STD MADE THE RMSE GO FROM 0.038 TO 0.036
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


randInds = randperm(length(symbolsUsed));
numTrain = floor(length(symbolsUsed)*0.9);
trainInds = randInds(1:numTrain);
testInds = randInds(numTrain+1:end);

Xtrain = XdataOther(trainInds,:);
Xtest = XdataOther(testInds,:);
Ytrain = Ydata(trainInds);
Ytest = Ydata(testInds);


save('TrainTestData7.mat','Xtrain','Xtest','Ytrain','Ytest');
%%

load('stockResults7.mat');

%rmse if random guess
%YdataRand = randn(size(Ydata)).*0.025;
%sqrt(sum((Ydata-YdataRand).^2)/length(Ydata))
%rmse of random guess: 0.0524
%rmse with xgboost: 0.03


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
%%
ii = find(yHatTest>0.01);
originalInds = goodInds(testInds(ii));

ii2 = find(yHatTest<-0.04);
origInds2 = goodInds(testInds(ii2));

moneyMade = 0;
moneySpent = 0;

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
moneyMade
moneySpent

ROI=moneyMade/moneySpent