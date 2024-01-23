import numpy as np
# import matplotlib.pyplot as plt


class Mean_Shift_Tracker:
    def __init__(self, x_center:int, y_center:int, obj_width:int, obj_height:int):
        # 目标框的中间
        self.prev_cx = x_center
        self.prev_cy = y_center
        self.curr_cx = x_center
        self.curr_cy = y_center
        
        # Bhattacharyya 系数
        self.prev_similarity_BC = 0.0
        self.curr_similarity_BC = 0.0
        
        # 保证目标高宽为奇数，方便处理
        if( obj_width %2 == 0 ):
            obj_width += 1    
        if( obj_height %2 ==0 ):
            obj_height +=1
            
        self.prev_width = obj_width 
        self.prev_height =  obj_height
        self.curr_width = obj_width
        self.curr_height = obj_height 
        
        # 模型参数定义
        self.bins_per_channel = 16
        self.bin_size = int( np.floor( 256 / self.bins_per_channel) )
        self.model_dim = self.bins_per_channel ** 3
        
        # 模型参数
        self.target_model:np.ndarray = np.zeros( self.model_dim )
        self.prev_model:np.ndarray = np.zeros( self.model_dim )
        self.curr_model:np.ndarray = np.zeros( self.model_dim )
        
        # 在颜色直方图中分配每个像素值的索引的数组
        self.combined_index:np.ndarray = np.zeros([self.curr_height , self.curr_width])
        self.max_itr = 5
        
        # 初始化模型核
        self.kernel_mask = self.init_kernel(self.curr_width, self.curr_height)
    
    def init_kernel(self, w, h):
        """ 
        初始化模型的掩码核
        """
        half_width = ( w -1 ) * 0.5
        half_height = ( h -1 ) * 0.5        
        x_limit = int( np.floor( ( w - 1)  * 0.5 ) )
        y_limit = int( np.floor( ( h -1 ) * 0.5 ) )
        x_range:np.ndarray = np.array( [ range( -x_limit , x_limit + 1 )])
        y_range:np.ndarray = np.array( [ range( -y_limit , y_limit + 1 ) ])
        y_range:np.ndarray = np.transpose(y_range)
        x_matrix:np.ndarray = np.repeat(x_range , y_limit*2+1 , axis=0)
        y_matrix:np.ndarray = np.repeat(y_range , x_limit*2+1 , axis=1)
        x_square:np.ndarray =  x_matrix ** 2
        y_square:np.ndarray =  y_matrix ** 2 
        x_square  = x_square / float( half_width * half_width ) 
        y_square  = y_square / float( half_height * half_height )
        kernel_mask:np.ndarray  = np.ones([h , w]) - (y_square + x_square)
        kernel_mask[kernel_mask<0] = 0
        
        return kernel_mask
    

    def update_target_model(self, ref_image):
        """利用输入图像计算初始追踪目标的模型"""
        self.update_object_model(ref_image)
        self.target_model = np.copy(self.curr_model)
    
    def update_object_model(self, image:np.ndarray):
        """根据输入图像更新颜色模型"""
        self.curr_model = self.curr_model * 0.0
        self.combined_index = self.combined_index * 0
        
        # 转换输入图像的数据类型
        image = image.astype( float )
        half_width = int( ( self.curr_width -1 ) * 0.5 )
        half_height = int( ( self.curr_height -1 ) * 0.5 )
        
        # 从图像中框选出检测框内的区域
        obj_image = image[self.curr_cy - half_height: self.curr_cy + half_height+1, 
                          self.curr_cx - half_width : self.curr_cx + half_width +1, :]

        # plt.figure()
        # plt.imshow(obj_image/256)
        # plt.show()

        # 对该区域的颜色进行建模
        index_matrix =  obj_image / self.bin_size  # 将色彩的深度降低至16位
        index_matrix =  np.floor(index_matrix).astype(int)
        b_index, g_index, r_index  = index_matrix[:,:,0], index_matrix[:,:,1], index_matrix[:,:,2], 
        # 生成位置色彩索引
        combined_index  =   b_index * np.power(self.bins_per_channel, 2) +\
                            self.bins_per_channel * g_index +\
                            r_index

        self.combined_index = combined_index.astype( int )
        if combined_index.shape != self.kernel_mask.shape:
            kernel_mask = self.init_kernel(combined_index.shape[1], combined_index.shape[0])
        else:
            kernel_mask = self.kernel_mask
        # print(combined_index.shape, self.kernel_mask.shape)
        # 更新颜色直方分布模型
        for i in range (self.curr_height):
            for j in range(self.curr_width):
                self.curr_model[ combined_index[ i , j ] ] += kernel_mask[i, j] 
        # 正则化
        sum_val = np.sum(self.curr_model)
        self.curr_model = self.curr_model / float(sum_val)
    

    def update_similarity_value(self):
        """ 
        计算并更新之前一帧与这一帧的BC距离
        """
        self.curr_similarity_BC  = 0.0
        # 计算两个分布间的BC相似度
        for i in range(self.model_dim):
            if(self.target_model[i] !=0 and self.curr_model[i] != 0 ):
                self.curr_similarity_BC += np.sqrt(self.target_model[i] * self.curr_model[i])
    

    def perform_mean_shift(self, image):
        """
        mean shift迭代
        """
        half_width = (self.curr_width -1) * 0.5
        half_height = (self.curr_height -1) * 0.5
        
        norm_factor= 0.0
        tmp_x = 0.0
        tmp_y = 0.0
        
        # 用上一轮的框中心初始化本轮的中心
        self.curr_cx = self.prev_cx
        self.curr_cy = self.prev_cy
        
        # 利用mean shift算法迭代更新
        for _ in range(self.max_itr):
            # 利用上一轮的目标位置检验本轮目标的置信度
            self.update_object_model(image)
            self.update_similarity_value()
            self.prev_similarity_BC = self.curr_similarity_BC
            feature_ratio = self.target_model / ( self.curr_model + 1e-5 )           
            
            # 计算新的目标位置
            for i in range (self.curr_height):
                for j in range(self.curr_width):     
                    tmp_x += (j - half_width) * feature_ratio[self.combined_index[ i , j ]]
                    tmp_y += (i - half_height) * feature_ratio[self.combined_index[ i , j ]]
                    norm_factor += feature_ratio[ self.combined_index[ i , j ] ]    

            mean_shift_x = tmp_x / norm_factor 
            mean_shift_y = tmp_y / norm_factor 
            
            # 利用 mean-shift 更新目标位置
            self.curr_cx += np.round(mean_shift_x) 
            self.curr_cy += np.round(mean_shift_y) 
            self.curr_cx = int(self.curr_cx)
            self.curr_cy = int(self.curr_cy)
            
            # 重新计算模型
            self.update_object_model(image)
            # 计算相似度
            self.update_similarity_value()
            # 微调搜索框的位置
            while(self.curr_similarity_BC - self.prev_similarity_BC < -0.01):
                self.curr_cx = int(np.floor((self.curr_cx + self.prev_cx) * 0.5))
                self.curr_cy = int(np.floor((self.curr_cy + self.prev_cy) * 0.5))
                # 检查模型是否收敛
                self.update_object_model(image)
                self.update_similarity_value()
                diff_x = self.prev_cx - self.curr_cx
                diff_y = self.prev_cy - self.curr_cy
                # 勾股定理计算更新前后，中心的距离
                euc_dist = np.power(diff_x , 2) + np.power(diff_y ,2)
                # 检查是否收敛
                if(euc_dist <= 2 ):
                    break
                
            diff_x = self.prev_cx - self.curr_cx
            diff_y = self.prev_cy - self.curr_cy
            
            # 再次微调
            euc_dist = np.power( diff_x , 2) + np.power( diff_y ,2 )
            
            self.prev_cx  = self.curr_cx
            self.prev_cy  = self.curr_cy

            # 检查收敛
            if( euc_dist <= 2 ):
                break
            
        