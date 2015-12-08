data = xlsread('train.csv');
train = data(1:600,:);
test = data(601:end,:);


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

bin = dec2bin(train(:,8),2); % binarizing the Embarkment feature(01:S,10:C,11:Q)
train(:,8)=bin(:,1);
train(:,9)=bin(:,2);
train(:,8)=floor(train(:,8)/49);
train(:,9)=floor(train(:,9)/49);

bin_t = dec2bin(test(:,8),2);
test(:,8)=bin_t(:,1);
test(:,9)=bin_t(:,2);
test(:,8)=floor(test(:,8)/49);
test(:,9)=floor(test(:,9)/49);

% normalizing age and fare features (ad hoc)
train(:,4)=train(:,4)/mean(train(:,4));
train(:,7)=train(:,7)/mean(train(:,7));

test(:,4)=test(:,4)/mean(test(:,4));
test(:,7)=test(:,7)/mean(test(:,7));

train(:,1)=[];
test(:,1)=[];

model_linear=svmtrain(label_train,train,'-c 1');
[yhat1,acc1,dec1]=svmpredict(label_test,test,model_linear);