clear all;
close all;
%%-----------------EE 660 ML Project 2015---------------------------------%%
%%--------------------Sagar Bachwani--------------------------------------%%
%%------------------------------------------------------------------------%%

data = xlsread('train.csv');
train = data(1:600,:);
test = data(601:end,:);

c=zeros(600,3);
d=zeros(291,3);
% converting passenger class categorical feature to binary


for u=1:600
    if train(u,2)==1
        c(u,1)=1;
        c(u,2)=0;
        c(u,3)=0;
    elseif train(u,2)==2
        c(u,1)=0;
        c(u,2)=1;
        c(u,3)=0;
    elseif train(u,2)==3
        c(u,1)=0;
        c(u,2)=0;
        c(u,3)=1;
    end
end

for u=1:291
    if test(u,2)==1
        d(u,1)=1;
        d(u,2)=0;
        d(u,3)=0;
    elseif test(u,2)==2
        d(u,1)=0;
        d(u,2)=1;
        d(u,3)=0;
    elseif test(u,2)==3
        d(u,1)=0;
        d(u,2)=0;
        d(u,3)=1;
    end
end

% replacing Nan values in age with the average age
avg = nanmean(train(:,4));
r = isnan(train(:,4));

for i=1:600
    if r(i,1) == 1
       train(i,4)=avg; 
    end
end

avg_t = nanmean(test(:,4));
t = isnan(test(:,4));

for s=1:291
    if t(s,1) ==1
        test(s,4)=avg_t;
    end
end

% randomly assigning Nan values in embarkment with value 2
q = isnan(train(:,8));
for i=1:600
    if q(i,1)==1
        train(i,8)=2;
    end
end

q_t = isnan(test(:,8));
for i=1:291
    if q_t(i,1)==1
        test(i,8)=2;
    end
end

train(:,8) = train(:,8) + 1; % integer adjustment
test(:,8) = test(:,8) + 1;

label_train = train(:,9); % separating the labels from train data
train(:,9)=[];
label_test = test(:,9); % separating the labels from test data
test(:,9)=[];

% Converting port of embarkment feature to binary
% Southampton: 1 0 0
% Cherbourg: 0 1 0
% Queenstown: 0 0 1

port=zeros(600,3);
for q=1:600
    if train(q,8)==1
        port(q,1)=1;
        port(q,2)=0;
        port(q,3)=0;
        
    elseif train(q,8)==2
        port(q,1)=0;
        port(q,2)=1;
        port(q,3)=0;
        
    elseif train(q,8)==3
        port(q,1)=0;
        port(q,2)=0;
        port(q,3)=1;
    end
end
train(:,8)=[];
train = [train port];

port2=zeros(291,3);
for q=1:291
    if test(q,8)==1
        port2(q,1)=1;
        port2(q,2)=0;
        port2(q,3)=0;
        
    elseif test(q,8)==2
        port2(q,1)=0;
        port2(q,2)=1;
        port2(q,3)=0;
        
    elseif test(q,8)==3
        port2(q,1)=0;
        port2(q,2)=0;
        port2(q,3)=1;
    end
end
test(:,8)=[];
test = [test port2];
        

% normalizing age and fare features (ad hoc)
%train(:,4)=train(:,4)/sum(train(:,4));
%train(:,7)=train(:,7)/sum(train(:,7));

%test(:,4)=test(:,4)/sum(test(:,4));
%test(:,7)=test(:,7)/sum(test(:,7));

train(:,[1 2])=[];
test(:,[1 2])=[];
train = [c train];
test = [d test];




% RANDOM FOREST using all data points in train set
forest = fitForest(train,label_train,'randomFeatures',2,'bagSize',1/3);
yhat1 = predictForest(forest,test);
err1 = 0;
for v=1:291
    if (label_test(v,1)~=yhat1(v,1))
        err1 = err1 + 1;
    end
end
disp(err1/291);


% nitr = 10;
% nntree = 30;
% errs_test = zeros(nitr,nntree);
% errs_train = zeros(nitr,nntree);
% r = randi(600,1,100);
% for ntree = 1:1:nntree
% 	ntree
%     
% 
% 	for itr = 1:nitr
% 		forest = fitForest(train(r,:),label_train(r,1),'randomFeatures',8,'bagSize',1/3,'ntrees',ntree);
% 		yhat_test = predictForest(forest,test);
% 		errs_test(itr,ntree) = mean(label_test ~= yhat_test);
%         
% 		yhat_train = predictForest(forest,train(r,:));
% 		errs_train(itr,ntree) = mean(label_train(r,1) ~= yhat_train);
% 	end
% end
% disp('finished');
% std_vs_ntree = std(errs_test,1);
% mean_vs_ntree_test = mean(errs_test,1);
% mean_vs_ntree_train = mean(errs_train,1);
% 
% %plot(std_vs_ntree,'g*');
% %hold on;
% plot(mean_vs_ntree_test,'r*');
% %hold on;
% %plot(mean_vs_ntree_train,'*');
% 
% title('mean and std of error rates of random forest');
% %legend('std of test error','mean test error', 'mean train error');
% xlabel('Number of Trees');
% 
% 
% 

% tf = fitctree(train(:,5),label_train);
% %treedisp(tf,'names',{'class1','class2','class3','sex','age','sibsp','parch','fare','S','C','Q'});
% % treedisp(tf,'name',{'sex','age'});
% view(tf,'Mode','graph');


tp = classregtree(train, label_train,'names',{'class1','class2','class3','sex','age','sibsp','parch','fare','S','C','Q'});
view(tp)
N = 600;
cp = cvpartition(label_train,'k',10);
dtclass = train;
%bad = ~strcmp(dtclass,label_train);
%dtResubErr = sum(bad) / N;

%cross validation

dtClassFun = @(train,label_train,test)(eval(classregtree(train,label_train),test));
dtCVErr  = crossval('mcr',train,label_train, ...
          'predfun', dtClassFun,'partition',cp);

disp(dtCVErr);


% LOGISTIC REGRESSION using MLE
model = logregFit(train, label_train, 'lambda', 0);
[yhat2, p] = logregPredict(model, test);
err2 = mean(yhat2 ~= label_test);
disp(err2);
w_mle_train = pinv(train)*label_train;

yhat_test = test*w_mle_train;












