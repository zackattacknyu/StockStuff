load('SAP500DataForOneYear.mat');

%training set will be Jan 10 - Jan 29 data

%target will be predicting whether closing price on February 5 2010
%   was higher or lower than closing price on January 29 2010

trainingInds = find(Date>20100110 & Date<20100200);
targetInds = find(Date==20100129 | Date==20100205);

symbols = unique(Symbol);
trainData = cell(1,length(symbols));
trainDataTarget = zeros(1,length(symbols));
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
    trainDataTarget(ii) = (targetData(2)>targetData(1));
    curTrain = zeros(length(curTrainInds),5);
    openPrice = Open(curTrainInds(1));
    curTrain(:,1)=Open(curTrainInds);
    curTrain(:,2)=Close(curTrainInds);
    curTrain(:,3)=Low(curTrainInds);
    curTrain(:,4)=High(curTrainInds);
    
    curTrain(:,1:4)=(curTrain(:,1:4)-openPrice)/openPrice;
    
    curTrain(:,5)=Volume(curTrainInds);
    trainData{ii} = curTrain;
end
%%
trainDataAdj = trainData(goodInds);
Xdata = zeros(length(trainDataAdj),numel(trainDataAdj{1}));
for i = 1:length(trainDataAdj)
    curFeat = trainDataAdj{i};
    Xdata(i,:) = curFeat(:)';
end
Ydata = trainDataTarget(goodInds);

save('TrainTestData1.mat','Xdata','Ydata');