function net = cnnsetup(net, x, y)
    inputmaps = 1;   %初始化网络输入层数为1层 
    mapsize = size(squeeze(x(1, :, :)));
  
%%=========================================================================  
% 主要功能：得到输入图像的行数和列数  
% 注意事项：1）B=squeeze(A) 返回和矩阵A相同元素但所有单一维都移除的矩阵B，单一维是满足size(A,dim)=1的维。  
%             train_x中图像的存放方式是三维的reshape(train_x',28,28,60000)，前面两维表示图像的行与列，  
%             第三维就表示有多少个图像。这样squeeze(x(:, :, 1))就相当于取第一个图像样本后，再把第三维  
%             移除，就变成了28x28的矩阵，也就是得到一幅图像，再size一下就得到了训练样本图像的行数与列数了  
%%=========================================================================  
    for l = 1 : numel(net.layers)   %  layer
        if strcmp(net.layers{l}.type, 's') %如果当前层是下采样层  
            mapsize = floor(mapsize / net.layers{l}.scale);
            for j = 1 : inputmaps
                net.layers{l}.b{j} = 0;
            end
        end
        %%=========================================================================  
        % 主要功能：获取下采样之后特征map的尺寸  
        % 注意事项：1）subsampling层的mapsize，最开始mapsize是每张图的大小28*28  
        %             这里除以scale=2，就是pooling之后图的大小，pooling域之间没有重叠，所以pooling后的图像为14*14  
        %             注意这里的右边的mapsize保存的都是上一层每张特征map的大小，它会随着循环进行不断更新  
        %%=========================================================================  
         %%=========================================================================  
       %如果当前层是卷基层  
        % 主要功能：获取卷积后的特征map尺寸以及当前层待学习的卷积核的参数数量  
        % 注意事项：1）旧的mapsize保存的是上一层的特征map的大小，那么如果卷积核的移动步长是1，那用  
        %             kernelsize*kernelsize大小的卷积核卷积上一层的特征map后，得到的新的map的大小就是下面这样  
        %          2）fan_out代表该层需要学习的参数个数。每张特征map是一个(后层特征图数量)*(用来卷积的patch图的大小)  
        %             因为是通过用一个核窗口在上一个特征map层中移动（核窗口每次移动1个像素），遍历上一个特征map  
        %             层的每个神经元。核窗口由kernelsize*kernelsize个元素组成，每个元素是一个独立的权值，所以  
        %             就有kernelsize*kernelsize个需要学习的权值，再加一个偏置值。另外，由于是权值共享，也就是  
        %             说同一个特征map层是用同一个具有相同权值元素的kernelsize*kernelsize的核窗口去感受输入上一  
        %             个特征map层的每个神经元得到的，所以同一个特征map，它的权值是一样的，共享的，权值只取决于  
        %             核窗口。然后，不同的特征map提取输入上一个特征map层不同的特征，所以采用的核窗口不一样，也  
        %             就是权值不一样，所以outputmaps个特征map就有（kernelsize*kernelsize+1）* outputmaps那么多的权值了  
        %             但这里fan_out只保存卷积核的权值W，偏置b在下面独立保存  
        %%=========================================================================  
        if strcmp(net.layers{l}.type, 'c')
            mapsize = mapsize - net.layers{l}.kernelsize + 1;
            fan_out = net.layers{l}.outputmaps * net.layers{l}.kernelsize ^ 2;
           %%=========================================================================  
            % 主要功能：获取卷积层与前一层输出map之间需要链接的参数链个数  
            % 注意事项：1）fan_out保存的是对于上一层的一张特征map，我在这一层需要对这一张特征map提取outputmaps种特征，  
            %             提取每种特征用到的卷积核不同，所以fan_out保存的是这一层输出新的特征需要学习的参数个数  
            %             而，fan_in保存的是，我在这一层，要连接到上一层中所有的特征map，然后用fan_out保存的提取特征  
            %             的权值来提取他们的特征。也即是对于每一个当前层特征图，有多少个参数链到前层  
            %%=========================================================================  
            for j = 1 : net.layers{l}.outputmaps  %  output map   %对于卷积层的每一个输出map  
                fan_in = inputmaps * net.layers{l}.kernelsize ^ 2;
                for i = 1 : inputmaps  %  input map  %对于上一层的每一个输出特征map（本层的输入map）  
                    net.layers{l}.k{i}{j} = (rand(net.layers{l}.kernelsize) - 0.5) * 2 * sqrt(6 / (fan_in + fan_out));
                end
                net.layers{l}.b{j} = 0;
            end
            inputmaps = net.layers{l}.outputmaps;
        end
    end
    fvnum = prod(mapsize) * inputmaps;
    onum = size(y, 1);

    net.ffb = zeros(onum, 1);
    net.ffW = (rand(onum, fvnum) - 0.5) * 2 * sqrt(6 / (onum + fvnum));
end
