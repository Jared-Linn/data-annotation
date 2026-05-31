import json
import sys

def extract_qa(input_json_path, output_txt_path):
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    lines = []
    for idx, item in enumerate(data):
        title = item.get('question_title', '').strip()
        question = item.get('question_content', '').strip()

        lines.append(f"question_title: {title}")
        lines.append(f"question_content: {question}")

        # 遍历所有回答，提取每个对话中的 content
        for answer in item.get('answers', []):
            for dialog in answer.get('dialogs', []):
                content = dialog.get('content', '').strip()
                if content:
                    lines.append(f"content: {content}")

        # 问题之间空一行（最后一个问题后不空）
        if idx != len(data) - 1:
            lines.append("")

    with open(output_txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✅ 处理完成，共 {len(data)} 条记录，输出文件：{output_txt_path}")

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("用法：python extract_qa.py <输入JSON文件> <输出TXT文件>")
        sys.exit(1)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    extract_qa(input_file, output_file)