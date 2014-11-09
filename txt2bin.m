% Convert text files into binary files

fid_text_TrainSet = fopen('.\\data\\training.txt','r');
fid_text_TrainLabels = fopen('.\\data\\training_label.txt','r');
fid_text_TestSet = fopen('.\\data\\testing.txt','r');
fid_bin_TrainSet = fopen('.\\data\\bin_training.dat','wb');
fid_bin_TrainLabels = fopen('.\\data\\bin_training_label.dat','wb');
fid_bin_TestSet = fopen('.\\data\\bin_testing.dat','wb');

while(~feof(fid_text_TrainSet))
    data = fscanf(fid_text_TrainSet,'%f',900);
    if(~isempty(data))
        fwrite(fid_bin_TrainSet,data,'unsigned char');
    else
        break;
    end
end

while(~feof(fid_text_TrainLabels))
    data = fscanf(fid_text_TrainLabels,'%f',1);
    if(~isempty(data))
        fwrite(fid_bin_TrainLabels,data,'unsigned char');
    else
        break;
    end
end

while(~feof(fid_text_TestSet))
    data = fscanf(fid_text_TestSet,'%f',900);
    if(~isempty(data))
        fwrite(fid_bin_TestSet,data,'unsigned char');
    else
        break;
    end
end
    
fclose('all');