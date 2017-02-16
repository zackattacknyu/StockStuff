load('SAP500DataForOneYear.mat');

%training set will be Jan 10 - Jan 29 data

%target will be predicting whether closing price on February 10 2010
%   was higher or lower than closing price on January 29 2010

trainingInds = find(Date>20100110 & Date<20100200);
targetInds = find(Date==20100129 | Date==20100210);

symbols = unique(Symbol);
trainData = cell(1,length(symbols));
trainDataTarget = zeros(1,length(symbols));
yDataCategory = zeros(1,length(symbols));
XtrainOther = zeros(length(symbols),11);
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
    if(currentY > 0.05)
        yDataCategory(ii)=1;
    elseif(currentY < -0.05)
        yDataCategory(ii)=-1;
    end
    curTrain = zeros(length(curTrainInds),5);
    openPrice = Open(curTrainInds(1));
    curTrain(:,1)=Open(curTrainInds);
    curTrain(:,2)=Close(curTrainInds);
    curTrain(:,3)=Low(curTrainInds);
    curTrain(:,4)=High(curTrainInds);
    
    curTrain(:,1:4)=(curTrain(:,1:4)-openPrice)/openPrice;
    
    curTrain(:,5)=Volume(curTrainInds);
    trainData{ii} = curTrain;
    
    XtrainOther(ii,1)=min(curTrain(:,1));
    XtrainOther(ii,2)=max(curTrain(:,1));
    XtrainOther(ii,3)=mean(curTrain(:,1));
    
    XtrainOther(ii,4)=min(curTrain(:,2));
    XtrainOther(ii,5)=max(curTrain(:,2));
    XtrainOther(ii,6)=mean(curTrain(:,2));
    
    XtrainOther(ii,7)=min(curTrain(:,5));
    XtrainOther(ii,8)=max(curTrain(:,5));
    XtrainOther(ii,9)=mean(curTrain(:,5));
    
    XtrainOther(ii,10)=min(curTrain(:,3));
    XtrainOther(ii,11)=max(curTrain(:,4));
end

XtrainOther = XtrainOther(goodInds,:);
trainDataAdj = trainData(goodInds);
Xdata = zeros(length(trainDataAdj),numel(trainDataAdj{1}));
for i = 1:length(trainDataAdj)
    curFeat = trainDataAdj{i};
    Xdata(i,:) = curFeat(:)';
end
Ydata = trainDataTarget(goodInds);
yDataCategory = yDataCategory(goodInds);

save('TrainTestData2.mat','XtrainOther','Xdata','Ydata','yDataCategory');
%%

%rmse if random guess
YdataRand = randn(size(Ydata)).*0.025;
sqrt(sum((Ydata-YdataRand).^2)/length(Ydata))
%rmse of random guess: 0.0524
%rmse with xgboost: 0.03

%continue with this trend 