import json
import uuid

# 输入文件和输出文件路径
input_file = 'E:/25-10-31的研究/数据集/nocold/nostart_reviews.json'
output_file = 'E:/25-10-31的研究/数据集/nocold/nostart_reviews1.json'

result = []

try:
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 移除行尾换行符并解析 JSON
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            # 构造新的结构
            # 注意：原数据没有 review_id，这里演示如何生成一个随机 ID
            item = {
                "center_user_id": data.get("center_user_id"),
                "user_id": data.get("user_id"),
                "business_name": data.get("business_name"),  # 模拟生成一个类似格式的 ID
                "business_id": data.get("business_id"),
                "categories": data.get("categories"),
                "stars": data.get("stars"),
                "date": data.get("date"),
                "text": data.get("text")
            }
            result.append(item)

    # 将整个数组保存为 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"转换完成！结果已保存至 {output_file}")

except FileNotFoundError:
    print("错误：找不到输入文件，请检查文件名。")