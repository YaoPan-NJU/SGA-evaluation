"""
方案解析脚本 - 从MD文件中提取JSON方案数据
已修复括号不平衡问题
"""
import json
import re
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "00_raw_md"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "03_schemes_data"
GROUPS = ["A", "B", "C", "D", "E", "F"]


def parse_md_file(filepath):
    """解析MD文件，提取所有方案"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    runs = re.split(r'## 运行结果 \d+:', content)[1:]
    results = []
    
    for run_text in runs:
        run_id_match = re.search(r'(RUN_\d+_\w+)', run_text)
        run_id = run_id_match.group(1) if run_id_match else 'Unknown'
        
        json_match = re.search(r'"final_proposal":\s*(\{.*\})', run_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(1).rstrip(',').strip()
            
            # 修复括号不平衡
            brace_count = 0
            for char in json_str:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
            
            if brace_count > 0:
                json_str = json_str + '}' * brace_count
            
            try:
                proposal = json.loads(json_str)
                results.append({
                    'run_id': run_id,
                    'proposal': proposal
                })
            except json.JSONDecodeError as e:
                print(f"  ✗ {run_id} JSON解析失败: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="从A-F组Markdown文件中提取方案JSON")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="包含 A.md 到 F.md 的目录，默认: 项目根目录/00_raw_md",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="输出JSON目录，默认: 项目根目录/03_schemes_data",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for group in GROUPS:
        print(f"\n解析 {group}.md...")
        filepath = input_dir / f"{group}.md"

        if filepath.exists():
            schemes = parse_md_file(filepath)
            print(f"  ✓ 成功解析 {len(schemes)} 个方案")

            output_file = output_dir / f"{group}_schemes.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(schemes, f, ensure_ascii=False, indent=2)
            print(f"  ✓ 已保存到 {output_file}")
        else:
            print(f"  ✗ 文件不存在: {filepath}")

    print("\n✅ 所有方案解析完成！")


if __name__ == "__main__":
    main()
