# -*- encoding: utf-8 -*-

# 数据预处理脚本
# 提取所有CONC文件中的数据，保存为字典
# 字典的格式为：{conc_name: {date_time: {'initial': initial_conc, 'final': final_conc}}}
# 其中conc_name为CONC文件的文件名，date_time为该数据对应的日期和时间，initial_conc为该数据对应的初始浓度，final_conc为该数据对应的最终浓度

import os
import pandas as pd
import re
import pickle as pkl
from tqdm import  tqdm
from multiprocessing import Pool

def process_file(args):
    conc_file, path_to_conc_files = args
    conc_name = conc_file.split(".")[0]
    conc_df = pd.read_csv(os.path.join(path_to_conc_files, conc_file))

    chemical_columns = [col for col in conc_df.columns if re.match(r'CONC\(\d+\)', col)]
    
    concentration_pairs = {}
    
    for i in range(len(conc_df) - 1):
        current_row = conc_df.iloc[i]
        current_jdate, current_jtime = current_row['JDATE'], current_row['JTIME']
        if int(current_jdate) == 2019001:  # skip if current_jdata is 2019001
        # if 2019182 <= int(current_jdate) <= 1019188:
            continue
        
        next_row = conc_df.iloc[i + 1]
        next_jdate, next_jtime = next_row['JDATE'], next_row['JTIME']
        
        if current_jdate == next_jdate and next_jtime == current_jtime:
            initial_conc = current_row[chemical_columns].values
            final_conc = next_row[chemical_columns].values
            concentration_pairs[(int(current_jdate), int(current_jtime))] = {'initial': initial_conc, 'final': final_conc}

    return conc_name, concentration_pairs

def extract_data_from_conc_file(path_to_conc_files, num_cpu=10):
    # find all files names start with CONC
    all_conc = [f for f in os.listdir(path_to_conc_files) if f.startswith("CONC")]
    # build the parameters for process_file
    all_conc_files = [(conc, path_to_conc_files) for conc in all_conc]
    with Pool(processes=num_cpu) as pool:
        results = list(tqdm(pool.imap(process_file, all_conc_files), total=len(all_conc_files)))
    merged_dataset = {name: pairs for name, pairs in results}
    return merged_dataset
    
if __name__ == "__main__":
    path_to_conc_files = "./data/raw_data/CONC"
    num_cpu = os.cpu_count() - 2
    merged_dataset = extract_data_from_conc_file(path_to_conc_files, num_cpu)
    # save 
    print("saving to ./data/processed_data/merged_dataset72all.pkl")
    with open("./data/processed_data/merged_dataset72all.pkl", "wb") as f:
        pkl.dump(merged_dataset, f)
    print("saving finshed")