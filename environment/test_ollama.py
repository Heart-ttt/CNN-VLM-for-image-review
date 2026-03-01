import ollama
import os
# 4G显存尽量关掉副屏，不然容易爆显存
def test_qwen_vision():
    # 1. 配置模型名称 (必须与截图一致)
    model_name = "qwen3-vl:2b"
    
    # 2. 配置图片路径 (请替换为你实际存在的图片路径)
    # 建议使用绝对路径，避免找不到文件的错误
    image_path = "./output/yolo11n_result.png" 
    
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"❌ 错误：找不到图片文件 -> {image_path}")
        print("💡 请修改代码中的 image_path 变量，指向一张真实的图片。")
        return

    print(f"✅ 准备就绪:")
    print(f"   模型：{model_name}")
    print(f"   图片：{image_path}")
    print("-" * 30)

    try:
        # 3. 发送请求
        # Qwen-VL 系列通常支持直接传图片路径
        stream = ollama.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': '这张图片里有什么？请详细描述。',
                'images': [image_path]
            }],
            stream=True,
            # 可选：增加超时时间，防止因首次加载慢而断开
            # timeout=120 
        )

        print("🤖 模型回答:\n")
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end='', flush=True)
        
        print("\n" + "-" * 30)
        print("✅ 测试成功完成！")

    except ollama.ResponseError as e:
        print(f"\n❌ Ollama 报错: {e.error}")
        if "not found" in str(e).lower():
            print(f"💡 提示：模型 '{model_name}' 未找到。请在终端运行：ollama pull {model_name}")
        elif "502" in str(e):
            print("💡 提示：502 错误通常是因为显存瞬间不足或模型加载冲突。")
            print("   1. 关闭 Ollama 客户端网页/应用，释放显存。")
            print("   2. 确保没有其他程序占用大量显存。")
            print("   3. 再次尝试运行。")
    except Exception as e:
        print(f"\n❌ 发生未知错误: {e}")

        
def load_prompt_from_txt(file_path: str) -> str:
    """
    从 txt 文件读取 Prompt 内容
    :param file_path: txt 文件的路径
    :return: 读取到的字符串内容
    """
    try:
        # encoding='utf-8' 非常重要，防止中文乱码
        with open(file_path, 'r', encoding='utf-8') as file:
            prompt_content = file.read()
        return prompt_content
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return None

"""# 使用示例
prompt_file = "safety_prompt.txt"  # 假设您的文件名
system_prompt = load_prompt_from_txt(prompt_file)"""
  

if __name__ == "__main__":
    test_qwen_vision()