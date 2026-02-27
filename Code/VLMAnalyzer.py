import os
import ollama


class VLM_Dec:
    def __init__(self, model_name: str, prompt_file_path: str = "LLMprompt.txt"):
        """
        初始化检测器
        
        :param model_name: Ollama 中已安装的 VLM 模型名称
        :param prompt_file_path: 提示词文件路径
        """
        self.model_name = model_name
        self.prompt = self.__load_prompt(prompt_file_path)

    def __load_prompt(self, file_path: str) -> str:
        """
        私有方法：读取提示词文件
        
        :param file_path: txt 文件路径
        :return: 提示词字符串
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"提示词文件不存在: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()

    def detect(self, image_path: str) -> str:
        """
        对外接口：检测图片是否违规
        
        :param image_path: 图片路径
        :return: 模型返回的检测结果字符串
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt,
                        "images": [image_path]
                    }
                ]
            )

            # 返回模型文本结果
            return response["message"]["content"]

        except Exception as e:
            return f"检测失败: {str(e)}"


if __name__ == "__main__":
    vlm = VLM_Dec("qwen3-VL:2b", "LLMprompt.txt")
    message = vlm.detect("../environment/image.png")   
    print(message) 