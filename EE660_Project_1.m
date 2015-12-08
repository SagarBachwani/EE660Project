clear all;
close all;
%%-----------------EE 660 ML Project 2015---------------------------------%%
%%--------------------Sagar Bachwani--------------------------------------%%
%%------------------------------------------------------------------------%%

%%--------Some preprocessing steps---------------------------------------%%
[num_tr,txt_tr,raw_tr] = xlsread('train.csv');
[num_ts,txt_ts,raw_ts] = xlsread('test.csv');
for i=2:892
    
    if (strcmp(raw_tr(i,5),'male'))
        raw_tr{i,5}=1;
    end
    
    if (strcmp(raw_tr(i,5),'female'))
        raw_tr{i,5}=0;
    end
end
    
% for i=2:892
%     
%     if (strcmp(raw_tr,'''S'''))
%         raw_tr{i,10}=0;
%     end
%     
%     if (strcmp(raw_tr,'''C'''))
%         raw_tr{i,10}=1;
%     end
%     
%     if (strcmp(raw_tr,'''Q'''))
%         raw_tr{i,10}=2;
%     end
%     
%     
% end

for i=2:419
    
    if (strcmp(raw_ts(i,3),'male'))
        raw_ts{i,3}=1;
    end
    
    if (strcmp(raw_ts(i,3),'female'))
        raw_ts{i,3}=0;
    end
end
    
for i=2:419
    
    if (strcmp(raw_ts{i,8},'S'))
        raw_ts{i,8}=0;
    end
    
    if (strcmp(raw_ts{i,8},'C'))
        raw_ts{i,8}=1;
    end
    
    if (strcmp(raw_ts{i,8},'Q'))
        raw_ts{i,8}=2;
    end
    
    
end

%raw_tr(:,4)=[];% discarding feature 'Name'
%raw_ts(:,3)=[];% discarding feature 'Name'

%raw_tr(:,12)=raw_tr(:,2);
%raw_tr(:,2)=[];% moving label to last column
%raw_tr(2,10)=strrep(raw_tr{2,10},'''S''','S');
%raw_tr(find(raw_tr(:,10)='''S'''))='S';
%s=categorical(raw_tr(:,10));


disp('Done')




