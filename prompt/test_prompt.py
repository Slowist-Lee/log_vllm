import csv
import sys

# 提升CSV字段大小限制，避免读取长文本时报错
csv.field_size_limit(sys.maxsize)

# 配置文件路径
csv_file_path = "./longbench-v2.csv"  # 源CSV文件
txt_save_path = "./long_prompt.txt"   # 目标TXT文件
# 定义需要拼接的列名列表
target_columns = ['question', 'choice_A', 'choice_B', 'choice_C', 'choice_D', 'answer', 'context']

try:
    # 打开CSV文件并按列名读取
    with open(csv_file_path, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # 读取第二行数据（第一行是列名）
        second_row = next(csv_reader, None)
    
    # 核心逻辑：拼接指定列 + 截取前10000个单词
    if second_row is not None:
        # 初始化拼接文本
        combined_text = ""
        
        # 遍历目标列，拼接内容（列之间用空格分隔，避免内容粘连）
        for col in target_columns:
            # 获取列内容，空值则用空字符串替代
            col_content = second_row.get(col, '').strip()
            # 拼接（非空内容才添加，避免多余空格）
            if col_content:
                combined_text += col_content + " "
        
        # 去除末尾多余的空格
        combined_text = combined_text.strip()
        
        # 处理拼接后为空的情况
        if not combined_text:
            print("警告：指定列的内容全部为空")
            final_content = ""
        else:
            # 按空格分割成单词列表
            words_list = combined_text.split()
            # 截取前10000个单词
            top_10000_words = words_list[:2048]
            # 重新拼接成文本
            final_content = ' '.join(top_10000_words)
            
            # 打印提示信息
            print(f"拼接后总单词数：{len(words_list)}，截取前10000个后：{len(top_10000_words)}")
    
    else:
        print("错误：CSV文件无第二行数据")
        final_content = ""
    
    # 将结果写入指定TXT文件
    with open(txt_save_path, 'w', encoding='utf-8') as txt_file:
        txt_file.write(final_content)
    
    print(f"操作完成！内容已保存到 {txt_save_path}")

except FileNotFoundError:
    print(f"错误：找不到CSV文件 {csv_file_path}")
except KeyError as e:
    print(f"错误：CSV文件中不存在 {e} 列，请检查列名是否正确（区分大小写）")
except Exception as e:
    print(f"处理文件时出错：{e}")