"""
从 03_schemes_data 重新生成评分任务队列。
"""
import argparse
import json
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GROUPS = ["A", "B", "C", "D", "E", "F"]


def load_group_schemes(schemes_dir: Path, group: str) -> list[dict]:
    scheme_file = schemes_dir / f"{group}_schemes.json"
    if not scheme_file.exists():
        raise FileNotFoundError(f"缺少方案数据文件: {scheme_file}")

    with open(scheme_file, "r", encoding="utf-8") as f:
        schemes = json.load(f)

    if not isinstance(schemes, list):
        raise ValueError(f"{scheme_file} 必须是JSON数组")

    return schemes


def build_task_queue(schemes_dir: Path, status: str) -> dict:
    tasks = []

    for group in GROUPS:
        schemes = load_group_schemes(schemes_dir, group)
        for index, scheme in enumerate(schemes, start=1):
            run_id = scheme.get("run_id")
            if not run_id:
                raise ValueError(f"{group}-{index:02d} 缺少 run_id")

            tasks.append({
                "id": f"{group}-{index:02d}",
                "group": group,
                "index": index,
                "status": status,
                "run_id": run_id,
            })

    return {
        "project_info": {
            "total_schemes": len(tasks),
            "groups": GROUPS,
            "schemes_per_group": 10,
            "rubric_version": "v1",
            "created_at": date.today().isoformat(),
            "source": "03_schemes_data",
        },
        "task_queue": tasks,
        "next_task": tasks[0]["id"] if tasks else None,
    }


def main():
    parser = argparse.ArgumentParser(description="从03_schemes_data生成01_task_queue.json")
    parser.add_argument(
        "--schemes-dir",
        type=Path,
        default=PROJECT_ROOT / "03_schemes_data",
        help="方案JSON目录，默认: 项目根目录/03_schemes_data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "01_task_queue.json",
        help="任务队列输出文件，默认: 项目根目录/01_task_queue.json",
    )
    parser.add_argument(
        "--status",
        choices=["pending", "completed", "failed"],
        default="pending",
        help="生成任务的初始状态，默认: pending",
    )
    args = parser.parse_args()

    queue = build_task_queue(args.schemes_dir.resolve(), args.status)

    with open(args.output.resolve(), "w", encoding="utf-8") as f:
        json.dump(queue, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"✅ 已生成 {len(queue['task_queue'])} 个任务: {args.output.resolve()}")


if __name__ == "__main__":
    main()
