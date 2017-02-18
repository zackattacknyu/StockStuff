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
    
    XdataOther(ii,4)=min(curTrain(:,2));
    XdataOther(ii,5)=max(curTrain(:,2));
    XdataOther(ii,6)=mean(curTrain(:,2));
    
    XdataOther(ii,7)=min(curTrain(:,5));
    XdataOther(ii,8)=max(curTrain(:,5));
    XdataOther(ii,9)=mean(curTrain(:,5));
    
    XdataOther(ii,10)=min(curTrain(:,3));
    XdataOther(ii,11)=max(curTrain(:,4));
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
%%

randInds = randperm(length(symbolsUsed));
numTrain = floor(length(symbolsUsed)*0.8);
trainInds = randInds(1:numTrain);
testInds = randInds(numTrain+1:end);

Xtrain = XdataOther(trainInds,:);
Xtest = XdataOther(testInds,:);
Ytrain = Ydata(trainInds);
Ytest = Ydata(testInds);
%%



%%

save('TrainTestData3.mat','Xtrain','Xtest','Ytrain','Ytest');
%%

%rmse if random guess
YdataRand = randn(size(Ydata)).*0.025;
sqrt(sum((Ydata-YdataRand).^2)/length(Ydata))
%rmse of random guess: 0.0524
%rmse with xgboost: 0.03

%%

[~,II] = sort(Ytest);

figure
hold on
plot(Ytest(II),'k-')
plot(yHatTest(II),'r-')
plot(zeros(1,100),'g--')
legend('Actual Stock Price Change','Predicted Stock Price Change');
hold off



%%
ii = find(yHatTest>0.03);
originalInds = goodInds(testInds(ii));

moneyMade = 0;

for jj = originalInds
    curSymbol = symbols(jj);
    symbolInds = find(strcmp(Symbol,curSymbol));
    curTargetInds = intersect(targetInds,symbolInds);
    if(length(curTrainInds)<1 || length(curTargetInds) < 2)
       continue 
    end
    targetData = Close(curTargetInds);
    moneyMade = moneyMade + (targetData(2)-targetData(1));
end
moneyMade