# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2023/11/07 19:13:41
@Author  :   Fei Gao
'''
import random
import torch
from torch.utils.data import Dataset
import numpy as np
def build_edges(reactions, species):
    # build hyperedges
    # structure: [[reactant set, products set]]

    hyperedges = []
    for k, v in reactions.items():
        reactants = [species.index(r) for r in v["reactants"]]
        products = [species.index(p) for p in v["products"]]
        hyperedges.append([set(reactants), set(products)])
        
        
    # 初始化边的列表
    reactants2reaction = []
    products2reaction = []
    reactions2reactants = []
    reactions2products = []

    # 构建边
    for reaction_id, (reactants_set, products_set) in enumerate(hyperedges):
        for reactant in reactants_set:
            reactants2reaction.append([reactant, reaction_id])
            reactions2reactants.append([reaction_id, reactant])
        for product in products_set:
            products2reaction.append([product, reaction_id])
            reactions2products.append([reaction_id, product])

    # 转换为tensor
    reactants2reaction_tensor = torch.tensor(reactants2reaction).t().contiguous()
    products2reaction_tensor = torch.tensor(products2reaction).t().contiguous()
    reactions2reactants_tensor = torch.tensor(reactions2reactants).t().contiguous()
    reactions2products_tensor = torch.tensor(reactions2products).t().contiguous()
    
    return reactants2reaction_tensor, products2reaction_tensor, reactions2reactants_tensor, reactions2products_tensor

# class ConcDataset(Dataset):
#     def __init__(self, all_data, dataset_type='train', val_days=0.3, test_days=1, min_max_values=None, **kwargs):
#         self.all_data = all_data
#         self.dataset_type = dataset_type
#         self.val_days = val_days
#         self.test_days = test_days
#         self.min_max_scale = kwargs.get('min_max_scale', False)
#         self.keys = self._get_keys()
#         print('Dataset: {}, number of data: {}'.format(self.dataset_type, len(self.keys)))
#         # TODO 训练集的min-max不能直接用到val和test！！！ 否则会造成数据泄露
#         if self.min_max_scale: self.__get_min_max__()
                
#         # #改
#         # if self.min_max_scale: 
#         #     self.__get_self_min_max__()
#         #     # print('Dataset: {},\n x_self_min: {}'.format(self.dataset_type, self.x_self_max))
#         #     if self.dataset_type == 'train':
#         #         self.__get_min_max__()
#         # if self.min_max_scale and min_max_values is not None:
#         #     self.x_min, self.x_max, self.y_min, self.y_max = min_max_values
            

#         self.env_min_max = {}  # 新增一个字典来保存每个物种的最小值和最大值
#         if self.min_max_scale: 
#             self.__get_self_min_max__()
#             # print('Dataset: {},\n x_self_min: {}'.format(self.dataset_type, self.x_self_max))
#             if self.dataset_type == 'train':
#                 self.__get_min_max__()
#         if self.min_max_scale and min_max_values is not None:
#             self.x_min, self.x_max, self.y_min, self.y_max = min_max_values      
            
class ConcDataset(Dataset):
    def __init__(self, all_data, dataset_type='train', val_days=0.3, test_days=1, min_max_values=None, **kwargs):
        self.all_data = all_data
        self.dataset_type = dataset_type
        self.val_days = val_days
        self.test_days = test_days
        self.min_max_scale = kwargs.get('min_max_scale', False)
        self.keys = self._get_keys()
        print('Dataset: {}, number of data: {}'.format(self.dataset_type, len(self.keys)))
        self.env_min_max = {}  # 新增一个字典来保存每个物种的最小值和最大值
        if self.min_max_scale: 
            self.__get_self_min_max__()
            # print('Dataset: {},\n x_self_min: {}'.format(self.dataset_type, self.x_self_max))
            if self.dataset_type == 'train':
                self.__get_min_max__()
        if self.min_max_scale and min_max_values is not None:
            self.x_min, self.x_max, self.y_min, self.y_max = min_max_values
             
  
        
    def _get_keys(self):
        # 获取所有数据的 date 和 time，然后根据 dataset_type 来分配键
        # test_days=1 表示 date 的最后一天都是测试集
        # val_days=0.1 表示 除了测试集，倒数第二天的随机10%是验证集
        # 其他的都是训练集
        all_date = {env: sorted(list(set([dates for (dates, times) in date_time_conc.keys()]))) for env, date_time_conc in self.all_data.items()}
        all_time = {env: sorted(list(set([times for (dates, times) in date_time_conc.keys()]))) for env, date_time_conc in self.all_data.items()}
        
        test_date = {env: dates[-self.test_days:] for env, dates in all_date.items()}
        val_date = {env: dates[-(self.test_days + 1):-self.test_days] for env, dates in all_date.items()}
        val_time = {}
        for env, times in all_time.items():
            val_size = int(len(times) * self.val_days)  # 计算验证集的大小
            val_indices = random.sample(range(len(times)), val_size)  # 随机选择索引
            val_time[env] = [times[i] for i in val_indices]  # 从时间列表中选择对应的时间点
        
        keys = []
        for env, date_time_conc in self.all_data.items():
            for (date, time), _ in date_time_conc.items():
                if self.dataset_type == 'train' and date not in test_date[env] and time not in val_time[env]:
                    keys.append((env, (date, time)))
                elif self.dataset_type == 'val' and date in val_date[env] and time in val_time[env]:
                    keys.append((env, (date, time)))
                elif self.dataset_type == 'test' and date in test_date[env]:
                    keys.append((env, (date, time)))                    
                elif self.dataset_type == 'all':
                    keys.append((env, (date, time)))        
        return keys

        

        
    
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        env_key, time_key = self.keys[idx]
        data = self.all_data[env_key][time_key]
        initial = torch.tensor(data['initial'], dtype=torch.float32)
        final = torch.tensor(data['final'], dtype=torch.float32)
        x, y = self.log_transform_scale_and_relative_change(initial, final)
        x, y = x.unsqueeze(1), y.unsqueeze(1)
        return x, y        

    
    
    
    def log_transform_scale_and_relative_change(self, a, b, inverse=False, epsilon=1e-25, offset=1e-2, lamda=0.1) -> tuple:
        """Scale data using min-max, log transform, and Z-score normalization, and handle inverse scaling."""
        if not inverse:
            initial_conc, final_conc = a, b
            delta_conc = final_conc - initial_conc
            x, y = initial_conc, delta_conc

            # Min-Max scaling if enabled
            if self.min_max_scale:
                # Clip values before scaling to avoid out-of-range values for log transformation
                x = np.clip(x, self.x_min, self.x_max)
                y = np.clip(y, self.y_min, self.y_max)
                print(f"裁剪后x max: {torch.max(x)}, x min: {torch.min(x)}")
                print(f"裁剪后y max: {torch.max(y)}, y min: {torch.min(y)}")
                x = (x - self.x_min) / (self.x_max - self.x_min + epsilon)
                y = (y - self.y_min) / (self.y_max - self.y_min + epsilon)
                print(f"mx后x max: {torch.max(x)}, x min: {torch.min(x)}")
                print(f"mx后y max: {torch.max(y)}, y min: {torch.min(y)}")
            # Log transform
            #x, y = torch.log10(x + epsilon), torch.log10(y + epsilon)
            x = torch.log10(x + 1 + epsilon)
            y = torch.log10(y + 1 + epsilon)
            # x = (x**lamda - 1) / lamda
            # y = (y**lamda - 1) / lamda            

            print(f"log缩放后x max: {torch.max(x)}, x min: {torch.min(x)}")
            print(f"log缩放后y max: {torch.max(y)}, y min: {torch.min(y)}")
            return x, y

        else:
            # x = np.zeros_like(a)  # Use appropriate shape based on a
            # y = np.zeros_like(b)
            # Inverse log transform
            #x, y = torch.pow(10, a) - epsilon, torch.pow(10, b) - epsilon
            print(f"Input a (scaled): {a}, b (scaled): {b}")
            # x = ((a * lamda) + 1)**(1 / lamda)
            # y = ((b * lamda) + 1)**(1 / lamda)
            x = torch.pow(10, a) - 1
            y = torch.pow(10, y) - 1            
            # print(f"反log缩放后After inverse log transform - Inverse x: {x}")
            # print(f"反log缩放后After inverse log transform - Inverse y: {y}")
            print(f"反log缩放后x max: {torch.max(x)}, x min: {torch.min(x)}")
            print(f"反log缩放后y max: {torch.max(y)}, y min: {torch.min(y)}")
            # Inverse Min-Max scaling if enabled
            if self.min_max_scale:
                x = (x * (self.x_max - self.x_min)) + self.x_min
                y = (y * (self.y_max - self.y_min)) + self.y_min
                print(f"反mx后x max: {torch.max(x)}, x min: {torch.min(x)}")
                print(f"反mx后y max: {torch.max(y)}, y min: {torch.min(y)}")
            # Calculate the final concentration from initial concentration and change
            final_conc = x + y
            return x, final_conc
        
#         if not inverse:        
#             # log10(x + 1) scaling
#             initial_conc, final_conc = a, b
#             delta_conc = final_conc - initial_conc
#             initial_conc, final_conc = torch.log10(initial_conc + 1), torch.log10(final_conc + 1)
#             # initial_conc, final_conc = torch.log10(initial_conc + epsilon), torch.log10(final_conc + epsilon)
#             x, y = initial_conc, final_conc - initial_conc
            
          
#             if self.min_max_scale:
#                 x = (x - self.x_min) / (self.x_max - self.x_min + epsilon)
#                 y = (y - self.y_min) / (self.y_max - self.y_min + epsilon)
#                 # x = (x**lamda - 1) / lamda
#                 # y = (y**lamda - 1) / lamda
#             return x, y
#         else:
#             x, y = a, b
#             if self.min_max_scale:
#                 x = x * (self.x_max - self.x_min + epsilon) + self.x_min
#                 y = y * (self.y_max - self.y_min + epsilon) + self.y_min
#                 # x = ((x * lamda) + 1)**(1 / lamda)
#                 # y = ((y * lamda) + 1)**(1 / lamda)
            
#             initial_conc, final_conc = x, x + y
#             initial_conc, final_conc = 10 ** initial_conc - 1, 10 ** final_conc - 1
#             # initial_conc, final_conc = 10 ** initial_conc - epsilon, 10 ** final_conc - epsilon
            
#             return x, final_conc         
        
   #原1.0版本     
    # def __get_min_max__(self):
    #     print('Getting min and max of initial and relative change for {} dataset...'.format(self.dataset_type))
    #     self.x_min = torch.tensor([float('inf')])
    #     self.x_max = torch.tensor([float('-inf')])
    #     self.y_min = torch.tensor([float('inf')])
    #     self.y_max = torch.tensor([float('-inf')])
    #     for env_key, time_key in self.keys:
    #         data = self.all_data[env_key][time_key]
    #         initial = torch.tensor(data['initial'], dtype=torch.float32)
    #         final = torch.tensor(data['final'], dtype=torch.float32)
    #         # 1. log10 scaling
    #         initial, final = torch.log10(initial + 1e-31), torch.log10(final + 1e-31)
    #         # 2. get the relative change
    #         x, y = initial, final - initial
    #         # 3. record the min and max of initial and and relative change
    #         self.x_min = torch.min(self.x_min, x.min())
    #         self.x_max = torch.max(self.x_max, x.max())
    #         self.y_min = torch.min(self.y_min, y.min())
    #         self.y_max = torch.max(self.y_max, y.max())        
    
    
    
    
    def __get_min_max__(self):
        print('获取 {} 数据集的初始值的最小和最大值...'.format(self.dataset_type))
        first_env_key, first_time_key = self.keys[0]
        first_data = self.all_data[first_env_key][first_time_key]
        num_species = len(first_data['initial'])
        self.x_min = torch.full((num_species,), float('inf'), dtype=torch.float32)
        self.x_max = torch.full((num_species,), float('-inf'), dtype=torch.float32)
        self.y_min = torch.full((num_species,), float('inf'), dtype=torch.float32)
        self.y_max = torch.full((num_species,), float('-inf'), dtype=torch.float32)    

        for env_key, time_key in self.keys:
            data = self.all_data[env_key][time_key]
            initial = torch.tensor(data['initial'], dtype=torch.float32)
            final = torch.tensor(data['final'], dtype=torch.float32)
            delta = final - initial
            # 更新最小值
            self.x_min = torch.min(self.x_min, initial)
            # 更新最大值
            self.x_max = torch.max(self.x_max, initial)
            
            self.y_min = torch.min(self.y_min, delta)
            # 更新最大值
            self.y_max = torch.max(self.y_max, delta)            
            
   
    def __get_self_min_max__(self):
        print('获取 {} 数据集的初始值的最小和最大值...'.format(self.dataset_type))
        # 初始化为正无穷大和负无穷大，大小为第一个数据点的物种数
        first_env_key, first_time_key = self.keys[0]
        first_data = self.all_data[first_env_key][first_time_key]
        num_species = len(first_data['initial'])
        self.x_self_min = torch.full((num_species,), float('inf'), dtype=torch.float32)
        self.x_self_max = torch.full((num_species,), float('-inf'), dtype=torch.float32)
        self.y_self_min = torch.full((num_species,), float('inf'), dtype=torch.float32)
        self.y_self_max = torch.full((num_species,), float('-inf'), dtype=torch.float32)
        for env_key, time_key in self.keys:
            data = self.all_data[env_key][time_key]
            initial = torch.tensor(data['initial'], dtype=torch.float32)
            final = torch.tensor(data['final'], dtype=torch.float32)
            delta = final - initial
            # 更新最小值
            self.x_self_min = torch.min(self.x_self_min, initial)
            # 更新最大值
            self.x_self_max = torch.max(self.x_self_max, initial)
            self.y_self_min = torch.min(self.y_self_min, delta)
            # 更新最大值
            self.y_self_max = torch.max(self.y_self_max, delta)                   
              
            
    
#     def log_transform_scale_and_relative_change(self, a, b, inverse=False, epsilon=1e-25) -> tuple:
#         """Scale data using min-max, log transform, and Z-score normalization, and handle inverse scaling."""
#         if not inverse:    
#             initial_conc, final_conc = a, b
            
#             # log10(x + 1) scaling
#             initial_conc, final_conc = torch.log2(initial_conc + 1), torch.log2(final_conc + 1)
#             # initial_conc, final_conc = torch.log10(initial_conc + epsilon), torch.log10(final_conc + epsilon)
#             x, y = initial_conc, final_conc - initial_conc
            
          
#             if self.min_max_scale:
#                 x = (x - self.x_min) / (self.x_max - self.x_min + epsilon)
#                 y = (y - self.y_min) / (self.y_max - self.y_min + epsilon)
#             return x, y
#         else:
#             x, y = a, b
#             if self.min_max_scale:
#                 x = x * (self.x_max - self.x_min + epsilon) + self.x_min
#                 y = y * (self.y_max - self.y_min + epsilon) + self.y_min
            
#             initial_conc, final_conc = x, x + y
#             initial_conc, final_conc = torch.pow(2, initial_conc) - 1, torch.pow(2, final_conc) - 1
#             # initial_conc, final_conc = 10 ** initial_conc - epsilon, 10 ** final_conc - epsilon
            
#             return x, final_conc
#     def log_transform_scale_and_relative_change(self, a, b, inverse=False, epsilon=1e-20) -> tuple:
#         """Scale data using min-max, log transform, and Z-score normalization, and handle inverse scaling."""        
        
#         if not inverse:
#             initial_conc, final_conc = a, b
#             delta_conc = final_conc - initial_conc
#             x, y = initial_conc, delta_conc

#             # 检查原始数据是否合理
#             # print(f"没缩放Before scaling - Initial concentrations (x): {x}")
#             # print(f"没缩放Before scaling - Change in concentrations (y): {y}")
#             print(f"没缩放x max: {torch.max(x)}, x min: {torch.min(x)}")
#             print(f"没缩放y max: {torch.max(y)}, y min: {torch.min(y)}")            
            
            
#             # Min-Max scaling if enabled
#             if self.min_max_scale:
#                 # Clip values before scaling to avoid out-of-range values for log transformation
#                 # x = np.clip(x, self.x_min, self.x_max)
#                 # y = np.clip(y, self.y_min, self.y_max)
#                 x = torch.clip(x, self.x_min, self.x_max)
#                 y = torch.clip(y, self.y_min, self.y_max)
# #                 print(f"clip的x max: {torch.max(x)}, x min: {torch.min(x)}")
# #                 print(f"clip的y max: {torch.max(y)}, y min: {torch.min(y)}")
#                 x = (x - self.x_min) / (self.x_max - self.x_min + epsilon)
#                 y = (y - self.y_min) / (self.y_max - self.y_min + epsilon)
#                 # x = x + epsilon
#                 # y = y + epsilon
                
# #                 print('Dataset: {},\n x_self_max: {}'.format(self.dataset_type, self.x_self_max))
# #                 print('Dataset: {},\n x_self_min: {}'.format(self.dataset_type, self.x_self_min))
# #                 print('Dataset: {},n x_self_max: {}'.format(self.dataset_type, torch.max(self.x_self_max)))
# #                 print('Dataset: {},n x_self_min: {}'.format(self.dataset_type, torch.min(self.x_self_min)))
# #                     # 检查缩放后的值是否合理
# #                 print(f"mx缩放后After scaling - Scaled x: {x}")
# #                 print(f"mx缩放后After scaling - Scaled y: {y}")
#                 print(f"mx缩放后x max: {torch.max(x)}, x min: {torch.min(x)}, Dataset: {self.dataset_type}")
#                 print(f"mx缩放后y max: {torch.max(y)}, y min: {torch.min(y)}")
#                 # assert torch.all(x >= 0) and torch.all(x <= 1), "Scaled x is out of range (0, 1)"
#                 # assert torch.all(y >= 0) and torch.all(y <= 1), "Scaled y is out of range (0, 1)"

            
#             # Log transform
#             x, y = torch.log10(x + 1), torch.log10(y + 1)
#             # 检查对数变换后的值
#             # print(f"log缩放后After log transform - Log transformed x: {x}")
#             # print(f"log缩放后After log transform - Log transformed y: {y}")
#             # print(f"log缩放后x max: {torch.max(x)}, x min: {torch.min(x)}")
#             # print(f"log缩放后y max: {torch.max(y)}, y min: {torch.min(y)}")
#             print(f"log缩放后x max: {torch.max(x)}, x min: {torch.min(x)}, Dataset: {self.dataset_type}")
#             print(f"log缩放后y max: {torch.max(y)}, y min: {torch.min(y)}")
#             return x, y

#         else:

#             # Inverse log transform
#             x, y = torch.pow(10, a) - 1, torch.pow(10, b) - 1
#             # x, y = torch.exp(a) - 1, torch.exp(b) - 1
#             # 检查反对数变换后的值
#             # print(f"反log缩放后After inverse log transform - Inverse x: {x}")
#             # print(f"反log缩放后After inverse log transform - Inverse y: {y}")
#             print(f"反log缩放后x max: {torch.max(x)}, x min: {torch.min(x)}")
#             print(f"反log缩放后y max: {torch.max(y)}, y min: {torch.min(y)}")
#             # Inverse Min-Max scaling if enabled
#             if self.min_max_scale:
#                 x = (x * (self.x_max - self.x_min)) + self.x_min
#                 y = (y * (self.y_max - self.y_min)) + self.y_min
#                 print('Dataset: {},n x_min: {}'.format(self.dataset_type, torch.min(self.x_min)))
#                 print('Dataset: {},n x_max: {}'.format(self.dataset_type, torch.min(self.x_max)))
#                 # x = x - epsilon
#                 # y = y - epsilon                
#                 # 检查反缩放后的值
#                 # print(f"反mx缩放后After inverse scaling - Final x: {x}")
#                 # print(f"反mx缩放后After inverse scaling - Final y: {y}")
#                 print(f"反mx缩放后x max: {torch.max(x)}, x min: {torch.min(x)}")
#                 print(f"反mx缩放后y max: {torch.max(y)}, y min: {torch.min(y)}")
#                 # print('Dataset: {},\n x_self_max: {}'.format(self.dataset_type, self.x_self_max))
#                 # print('Dataset: {},\n x_self_min: {}'.format(self.dataset_type, self.x_self_min))
#                 print('Dataset: {},n x_self_min: {}'.format(self.dataset_type, torch.min(self.x_self_min)))
#                 print('Dataset: {},n x_self_min: {}'.format(self.dataset_type, torch.min(self.x_self_max)))                
#                 # assert torch.all(x >= self.x_min) and torch.all(x <= self.x_max), "Inverse scaled x is out of range"
#                 # assert torch.all(y >= self.y_min) and torch.all(y <= self.y_max), "Inverse scaled y is out of range"

#             # Calculate the final concentration from initial concentration and change
#             final_conc = x + y
#             print(f"最终浓度: {torch.max(final_conc), torch.min(final_conc)}")
#             return x, final_conc            
            


    
                        
# dnn版本数据缩放代码            
#         if not inverse:

#             initial_conc, final_conc = a, b
#               #改
#             delta_conc = final_conc - initial_conc
#             x, y = initial_conc, delta_conc    
#             # Min-Max缩放
#             if self.min_max_scale:
#                 x = torch.clamp(x, self.x_min, self.x_max)
#                 y = torch.clamp(y, self.y_min, self.y_max)
#                 x = (x - self.x_min) / (self.x_max - self.x_min + epsilon)
#                 y = (y - self.y_min) / (self.y_max - self.y_min + epsilon)    
#             #print('Min-Max缩放')
#     # Log变换
#             x, y = torch.log10(x + 1), torch.log10(y + 1)
#             #print('进行Log变换')
#             return x, y
#         else:
#             print('进行逆Log变换')# 逆Log变换
#             x, y = torch.pow(10, a) - 1, torch.pow(10, b) - 1

#             # 逆Min-Max缩放
#             if self.min_max_scale:
#                 x = x * (self.x_max - self.x_min) + self.x_min
#                 y = y * (self.y_max - self.y_min) + self.y_min

#             final_conc = x + y
#             print('逆Min-Max缩放')
#             return x, final_conc   

        
#         if not inverse:
#             # 正向缩放：先进行Log变换，再进行Min-Max缩放
#             initial_conc, final_conc = a, b
#             delta_conc = final_conc - initial_conc
#             x, y = initial_conc, delta_conc

#             # Log变换
#             x, y = torch.log10(x + 1e-1), torch.log10(y + 1e-1)  # 注意：对数变换应先进行

#             # Min-Max缩放
#             if self.min_max_scale:
#                 # Clamp 限制范围
#                 x = torch.clamp(x, self.x_min, self.x_max)
#                 y = torch.clamp(y, self.y_min, self.y_max)
#                 # 归一化 Min-Max
#                 x = (x - self.x_min) / (self.x_max - self.x_min + epsilon)
#                 y = (y - self.y_min) / (self.y_max - self.y_min + epsilon)

#             return x, y

#         else:
#             print('进行逆缩放')  # 逆缩放：先进行逆Min-Max缩放，再进行逆Log变换

#             # 逆Min-Max缩放
#             x, y = a, b
#             if self.min_max_scale:
#                 x = x * (self.x_max - self.x_min) + self.x_min
#                 y = y * (self.y_max - self.y_min) + self.y_min

#             # 逆Log变换
#             x = torch.pow(10, x) - 1e-1
#             y = torch.pow(10, y) - 1e-1

#             final_conc = x + y
#             return x, final_conc
        
        # if self.min_max_scale and not hasattr(self, 'x_min'):
        #     raise ValueError("Min-Max values are not initialized. Call __get_min_max__() first.") 
        
        
        
        

        
        
#         #此种方式中，用epsilon-24的效果显然比-30好，但似乎+1改成+0.5模型训练迭代次数非常多且存在大幅度跳跃，可能有过拟合的情况； 使用先log+1再lamda的方法结果：
#         if not inverse:        
#             # log10(x + 1) scaling
#             initial_conc, final_conc = a, b
#             delta_conc = final_conc - initial_conc
#             # initial_conc, final_conc = torch.log10(initial_conc + 1), torch.log10(final_conc + 1)
#             # initial_conc, final_conc = torch.log10(initial_conc + epsilon), torch.log10(final_conc + epsilon)
#             x, y = initial_conc, final_conc - initial_conc
            
          
#             if self.min_max_scale:
#                 # x = (x - self.x_min) / (self.x_max - self.x_min + epsilon)
#                 # y = (y - self.y_min) / (self.y_max - self.y_min + epsilon)
#                 x = (x**lamda - 1) / lamda
#                 y = (y**lamda - 1) / lamda
#             return x, y
#         else:
#             x, y = a, b
#             if self.min_max_scale:
#                 # x = x * (self.x_max - self.x_min + epsilon) + self.x_min
#                 # y = y * (self.y_max - self.y_min + epsilon) + self.y_min
#                 x = ((x * lamda) + 1)**(1 / lamda)
#                 y = ((y * lamda) + 1)**(1 / lamda)
            
#             initial_conc, final_conc = x, x + y
#             # initial_conc, final_conc = 10 ** initial_conc - 1, 10 ** final_conc - 1
#             # initial_conc, final_conc = 10 ** initial_conc - epsilon, 10 ** final_conc - epsilon
            
#             return x, final_conc        
        
        
        
        



        

            
    
