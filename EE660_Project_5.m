clear all;
close all;

data = xlsread('train.csv');
train = data(1:600,:);
test = data(601:end,:);

c=zeros(600,3);
d=zeros(291,3);

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
train(:,4)=train(:,4)/mean(train(:,4));
train(:,7)=train(:,7)/mean(train(:,7));

test(:,4)=test(:,4)/mean(test(:,4));
test(:,7)=test(:,7)/mean(test(:,7));

train(:,[1 2])=[];
test(:,[1 2])=[];
train = [c train];
test = [d test];



w_mle_train = pinv(train)*label_train;

yhat_test = test*w_mle_train;

for g=1:291
    if yhat_test(g,1)<0.5
        yhat_test(g,1)=0;
    end
end

mse = (uint8(yhat_test(:,1)) - uint8(label_test(:,1)));
error = sum(mse)/291;




        


