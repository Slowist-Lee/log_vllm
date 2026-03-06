#!/usr/bin/env python3
import time
from modelscope import snapshot_download

def download_with_retry():
    model_id = 'LLM-Research/Meta-Llama-3-8B-Instruct'
    local_path = './llama3_8b_model'
    max_retries = 1000  # 设置一个极大的重试次数
    
    print(f"开始从 ModelScope 下载 {model_id}...")
    print("提示：自带断点续传功能，如遇网络中断将自动重试。\n")

    for attempt in range(max_retries):
        try:
            # snapshot_download 本身会检查本地缓存，实现断点续传
            model_dir = snapshot_download(model_id, local_dir=local_path)
            print(f"\n✅ 模型完整下载成功！保存在: {model_dir}")
            break  # 下载成功，跳出循环
            
        except Exception as e:
            print(f"\n❌ 第 {attempt + 1} 次尝试中断。错误信息: {e}")
            print("休眠 5 秒后继续进行断点续传...")
            time.sleep(5)

if __name__ == "__main__":
    download_with_retry()

