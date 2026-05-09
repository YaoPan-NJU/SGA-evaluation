"""
API评分脚本 - 调用小米mimo-v2.5-pro进行深度化学推理评分
"""
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMES_DATA_DIR = PROJECT_ROOT / "03_schemes_data"
RESULTS_DIR = PROJECT_ROOT / "04_scoring_results"
TASK_QUEUE_FILE = PROJECT_ROOT / "01_task_queue.json"

DIMENSIONS = ["D1", "D2", "D3", "D4", "D5", "D6"]
WEIGHTS = {"D1": 25, "D2": 25, "D3": 20, "D4": 10, "D5": 10, "D6": 10}

# 加载项目根目录下的.env文件（如果存在）
load_dotenv(PROJECT_ROOT / ".env")

# API配置
API_BASE_URL = os.environ.get("MIMO_API_BASE_URL", "https://token-plan-ams.xiaomimimo.com/v1")
API_MODEL = os.environ.get("MIMO_API_MODEL", "mimo-v2.5-pro")
API_KEY = os.environ.get("MIMO_API_KEY", "")
API_TEMPERATURE = float(os.environ.get("MIMO_API_TEMPERATURE", "0.3"))
API_MAX_TOKENS = int(os.environ.get("MIMO_API_MAX_TOKENS", "4000"))
API_INTERVAL_SECONDS = float(os.environ.get("MIMO_API_INTERVAL_SECONDS", "2"))


def format_steps(steps: list[dict]) -> str:
    if not steps:
        return "无"

    lines = []
    for step in steps:
        step_no = step.get("step", "N/A")
        description = step.get("description", "")
        temperature = step.get("temperature", "N/A")
        duration = step.get("duration", "N/A")
        lines.append(
            f"Step {step_no}: {description}\n"
            f"  - 温度: {temperature}\n"
            f"  - 时间: {duration}"
        )
    return "\n".join(lines)


def format_mapping(mapping: dict) -> str:
    if not mapping:
        return "无"
    return "\n".join(f"- {key}: {value}" for key, value in mapping.items())


def format_list(items: list) -> str:
    if not items:
        return "无"
    return "\n".join(f"- {item}" for item in items)


def build_scoring_prompt(scheme_data: Dict[str, Any], scheme_id: str) -> str:
    """
    构建详细的化学推理评分prompt。
    """
    proposal = scheme_data["proposal"]
    run_id = scheme_data["run_id"]

    steps = proposal.get("steps", [])
    raw_materials = proposal.get("raw_materials", [])
    key_parameters = proposal.get("key_parameters", {})
    equipment = proposal.get("equipment_required") or proposal.get("equipment", [])
    safety_notes = proposal.get("safety_notes", [])
    rationale = proposal.get("rationale", "")

    prompt = f"""你是一个专业的材料化学专家，请对以下材料合成方案进行严格的李克特量表评分（v1版本，6维度加权评分）。

## 方案信息
**方案ID**: {scheme_id}
**Run ID**: {run_id}

## 合成路线设计理由
{rationale if rationale else '无'}

## 合成步骤
{format_steps(steps)}

## 原料清单
{format_list(raw_materials)}

## 关键参数
{format_mapping(key_parameters)}

## 设备要求
{format_list(equipment)}

## 安全注意事项
{format_list(safety_notes)}

## 评分任务

请严格按照以下6个维度进行**深度化学推理**评分，每个维度1-10分：

### D1: 设计到合成的转化准确性 (权重25%)
**评估标准**：
- 9-10分：载体、复合组分、活性位点、形貌、表面特征都有明确合成来源
- 7-8分：主要载体和官能团位点可达；次要结构细节需确认
- 5-6分：可获得相关材料，但几个设计特征缺乏明确的形成路径
- 3-4分：仅能获得大致相似的材料；关键设计特征缺失
- 1-2分：设计与方案明显不匹配

**推理要求**：
1. 识别方案声称要合成的材料（从rationale和步骤推断）
2. 对每个设计特征（载体、复合组分、官能团、形貌），在steps中找到对应的合成路径
3. 评估每个步骤是否真正能生成目标结构（化学上合理吗？）
4. 检查是否所有设计特征都有形成路径

### D2: 反应和工艺参数可行性 (权重25%)
**评估标准**：
- 9-10分：反应化学成熟；前驱体兼容性、工艺参数、后处理条件均有充分论证
- 7-8分：主要反应可能成功，只需有限的条件优化
- 5-6分：化学有合理基础，但接枝效率、副反应或固定稳定性不确定
- 3-4分：关键反应或工艺条件存疑，实验失败风险高
- 1-2分：关键反应内在不一致或基本不可行

**推理要求**：
1. 识别所有关键化学反应（共沉淀、酯化、酰胺化、环氧化、季铵化等）
2. 分析每个反应的条件是否支持该反应机理：
   - pH值是否合理？（如季铵化需要碱性使氨基去质子化）
   - 温度是否合适？（如Fe3O4共沉淀通常70-90°C）
   - 试剂添加顺序正确吗？
3. 评估副反应风险（如ECH在强碱下易水解）
4. 检查前驱体兼容性（不同步骤的试剂会相互干扰吗？）

### D3: 结构控制和质量控制完整性 (权重20%)
**评估标准**：
- 9-10分：结构控制明确，并有适当的表征方法和定量验收标准支撑
- 7-8分：主要结构特征有控制策略和表征方法；部分标准未定量
- 5-6分：提供了一些结构控制逻辑，但QC指标仍为通用描述
- 3-4分：结构声明大多未经可验证的控制路线证实
- 1-2分：目标结构、路线和表征计划脱节

**推理要求**：
1. 列出方案声称的结构特征
2. 检查表征方法是否适合验证所声称的结构：
   - FTIR能检测什么官能团？
   - XRD能验证什么结晶相？
   - BET能确认什么孔隙特征？
3. 评估验收标准的定量性（如"BET > 500 m²/g"是定量，"确认成功"是定性）
4. 检查是否有中间QC检查点

### D4: 批次稳定性和实验可重复性 (权重10%)
**评估标准**：
- 9-10分：参数、关键控制点和中间检查充分定义，具有强批次可重复性
- 7-8分：主要参数清晰，方案通常可重复
- 5-6分：路线可执行，但一些关键参数或工艺窗口未充分说明
- 3-4分：多个步骤依赖隐性经验，造成显著的批次间差异
- 1-2分：关键参数缺失，使重复不切实际

**推理要求**：
1. 检查所有试剂是否有精确用量（g/mL/molarity）
2. 检查关键参数（温度、时间、pH、压力）是否都有明确数值
3. 识别关键控制点（如"缓慢滴加防止局部过热"）
4. 评估参数窗口是否足够窄以保证批次一致性

### D5: 路线简洁性和方法适当性 (权重10%)
**评估标准**：
- 9-10分：路线使用足够简单适当的方法，设备负担低，放大潜力好
- 7-8分：路线复杂度中等，但复杂性主要由目标结构论证
- 5-6分：路线偏复杂，某些步骤可简化或替换
- 3-4分：路线过度工程化；高温、高压、惰性气氛或专用步骤缺乏论证
- 1-2分：路线复杂度与目标材料严重不匹配

**推理要求**：
1. 评估步骤数量是否合理（简单材料3-5步，复合材料5-8步）
2. 检查是否有过度复杂化（能否合并步骤？）
3. 评估复杂操作的正当性（如惰性气氛是否因为试剂易氧化？）
4. 评估设备要求是否合理（常规实验室能否实现？）

### D6: 安全性、试剂可获得性和风险缓解 (权重10%)
**评估标准**：
- 9-10分：主要使用易得试剂和可控条件；安全预防措施和废物处理明确指定
- 7-8分：存在常见危害但可控，有充分的保护和废物处理措施
- 5-6分：路线涉及高温、高压、腐蚀性试剂、有机溶剂或中等毒性试剂，需严格管理
- 3-4分：存在高毒性、强氧化性、重金属、含氟废物或管制化学品风险，缓解不足
- 1-2分：安全或合规风险超出标准实验室合理管理能力

**推理要求**：
1. 识别所有主要风险（腐蚀性、毒性、易燃性、氧化性、反应性）
2. 评估缓解措施是否充分且具体（不能只说"在通风橱操作"）
3. 检查废物处理方案是否合理
4. 评估试剂是否容易获取

## 输出格式要求

请严格按以下JSON格式输出评分结果（不要输出其他内容）：

```json
{{
  "scores": {{
    "D1": <整数1-10>,
    "D2": <整数1-10>,
    "D3": <整数1-10>,
    "D4": <整数1-10>,
    "D5": <整数1-10>,
    "D6": <整数1-10>
  }},
  "reasons": {{
    "D1": "<具体理由，必须引用具体步骤编号和参数，说明化学原理，1-3句话>",
    "D2": "<具体理由，必须引用具体步骤编号和参数，说明化学原理，1-3句话>",
    "D3": "<具体理由，必须引用具体步骤编号和参数，说明化学原理，1-3句话>",
    "D4": "<具体理由，必须引用具体步骤编号和参数，说明化学原理，1-3句话>",
    "D5": "<具体理由，必须引用具体步骤编号和参数，说明化学原理，1-3句话>",
    "D6": "<具体理由，必须引用具体步骤编号和参数，说明化学原理，1-3句话>"
  }}
}}
```

**重要要求**：
1. 每个维度的理由必须引用具体步骤编号（如"Step 2"或"步骤2"）
2. 每个维度的理由必须说明化学原理（如"碱性条件使氨基去质子化促进SN2反应"）
3. 不能使用"反应化学成熟"、"工艺参数合理"等通用套话
4. 分数必须严格对照量表的5级描述
5. 不要照抄既有文字；必须基于本方案的步骤、参数、原料和安全信息逐项推理
6. 只输出JSON，不要输出其他内容
"""
    return prompt


def extract_json(content: str) -> Dict[str, Any]:
    """从模型响应中提取JSON对象。"""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL | re.IGNORECASE)
    if fenced:
        return json.loads(fenced.group(1))

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("响应中未找到JSON对象", content, 0)

    return json.loads(content[start:end + 1])


def call_mimo_api(prompt: str, max_retries: int = 3) -> Dict[str, Any]:
    """
    调用小米mimo API，并解析JSON响应。
    """
    if not API_KEY:
        raise ValueError("MIMO_API_KEY环境变量未设置，请先设置API密钥")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": API_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "你是材料化学专家，擅长材料合成方案的化学推理和评估。请严格按照用户要求逐项分析，只输出JSON格式的结果。",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": API_TEMPERATURE,
        "max_tokens": API_MAX_TOKENS,
    }

    last_content = ""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{API_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120,
            )
            response.raise_for_status()

            api_result = response.json()
            last_content = api_result["choices"][0]["message"]["content"]
            return extract_json(last_content)

        except requests.exceptions.RequestException as e:
            print(f"  ⚠️ API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
            else:
                raise
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  ⚠️ 解析API响应失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if last_content:
                print(f"  原始内容片段: {last_content[:300]}...")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise

    raise RuntimeError(f"API调用失败：已重试{max_retries}次")


def calculate_weighted_total(scores: Dict[str, int]) -> float:
    return round(sum(WEIGHTS[dim] * (scores[dim] / 10) for dim in DIMENSIONS), 2)


def calculate_grade(weighted_total: float) -> str:
    if weighted_total >= 85:
        return "A"
    if weighted_total >= 75:
        return "B+"
    if weighted_total >= 65:
        return "B"
    if weighted_total >= 55:
        return "C"
    return "D"


def has_step_reference(reason: str) -> bool:
    return bool(re.search(r"\bStep\s*\d+\b|步骤\s*\d+", reason, re.IGNORECASE))


def validate_and_normalize_result(result: Dict[str, Any], scheme_id: str) -> Dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("API返回结果不是JSON对象")
    if not isinstance(result.get("scores"), dict):
        raise ValueError("API返回数据格式错误：缺少scores对象")
    if not isinstance(result.get("reasons"), dict):
        raise ValueError("API返回数据格式错误：缺少reasons对象")

    normalized_scores = {}
    normalized_reasons = {}

    for dim in DIMENSIONS:
        if dim not in result["scores"]:
            raise ValueError(f"scores缺少{dim}")
        score = result["scores"][dim]
        if isinstance(score, float) and score.is_integer():
            score = int(score)
        if not isinstance(score, int) or not 1 <= score <= 10:
            raise ValueError(f"{dim}分数必须是1-10整数，当前值: {score!r}")
        normalized_scores[dim] = score

        reason = result["reasons"].get(dim)
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"reasons缺少{dim}或理由为空")
        if not has_step_reference(reason):
            raise ValueError(f"{dim}理由未引用具体步骤编号")
        normalized_reasons[dim] = reason.strip()

    weighted_total = calculate_weighted_total(normalized_scores)
    grade = calculate_grade(weighted_total)

    result["scores"] = normalized_scores
    result["reasons"] = normalized_reasons
    result["weighted_total"] = weighted_total
    result["grade"] = grade
    result["calculation"] = (
        f"25×{normalized_scores['D1']/10} + "
        f"25×{normalized_scores['D2']/10} + "
        f"20×{normalized_scores['D3']/10} + "
        f"10×{normalized_scores['D4']/10} + "
        f"10×{normalized_scores['D5']/10} + "
        f"10×{normalized_scores['D6']/10} = {weighted_total}"
    )

    print(f"  ✓ {scheme_id} 返回结构、分数范围、步骤引用、总分和等级校验通过")
    return result


def save_task_queue(task_queue: Dict[str, Any]) -> None:
    tmp_file = TASK_QUEUE_FILE.with_suffix(".json.tmp")
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(task_queue, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_file.replace(TASK_QUEUE_FILE)


def update_next_task(task_queue: Dict[str, Any]) -> None:
    next_task = next((t["id"] for t in task_queue["task_queue"] if t["status"] == "pending"), None)
    task_queue["next_task"] = next_task


def score_single_scheme(scheme_data: Dict[str, Any], scheme_id: str, output_dir: Path) -> Dict[str, Any] | None:
    """
    对单个方案进行评分。
    """
    print(f"\n{'=' * 60}")
    print(f"评分 {scheme_id}...")
    print("=" * 60)

    try:
        prompt = build_scoring_prompt(scheme_data, scheme_id)
        result = call_mimo_api(prompt)
        result = validate_and_normalize_result(result, scheme_id)

        group = scheme_id.split("-")[0]
        index = int(scheme_id.split("-")[1])
        result["scheme_id"] = scheme_id
        result["run_id"] = scheme_data["run_id"]
        result["group"] = group
        result["index"] = index
        result["scored_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        result["scorer_notes"] = f"深度化学推理评分 - {API_MODEL} API"

        group_dir = output_dir / f"{group}_group"
        group_dir.mkdir(parents=True, exist_ok=True)

        output_file = group_dir / f"{scheme_id}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")

        print(f"✅ {scheme_id} 评分完成: 总分={result['weighted_total']}, 等级={result['grade']}")
        return result

    except Exception as e:
        print(f"❌ {scheme_id} 评分失败：{type(e).__name__}: {e}")
        traceback.print_exc()
        return None


def load_scheme_for_task(task: Dict[str, Any]) -> Dict[str, Any]:
    group = task["group"]
    index = task["index"]

    scheme_file = SCHEMES_DATA_DIR / f"{group}_schemes.json"
    with open(scheme_file, "r", encoding="utf-8") as f:
        schemes = json.load(f)

    if not 1 <= index <= len(schemes):
        raise IndexError(f"{task['id']} 方案数据不存在")

    scheme_data = schemes[index - 1]
    actual_run_id = scheme_data.get("run_id")
    if task.get("run_id") != actual_run_id:
        raise ValueError(
            f"{task['id']} run_id不匹配：队列={task.get('run_id')}，方案数据={actual_run_id}"
        )

    return scheme_data


def main():
    """主函数：批量评分"""
    with open(TASK_QUEUE_FILE, "r", encoding="utf-8") as f:
        task_queue = json.load(f)

    pending_tasks = [t for t in task_queue["task_queue"] if t["status"] == "pending"]
    print(f"\n📋 待评分任务: {len(pending_tasks)} 个")

    if not pending_tasks:
        print("✅ 没有pending任务。若需重跑，请先重置任务队列。")
        return

    completed = 0
    failed = 0

    for task in pending_tasks:
        scheme_id = task["id"]

        try:
            scheme_data = load_scheme_for_task(task)
            result = score_single_scheme(scheme_data, scheme_id, RESULTS_DIR)

            if result:
                task["status"] = "completed"
                task["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                task["result_file"] = str(
                    RESULTS_DIR / f"{task['group']}_group" / f"{scheme_id}.json"
                )
                task.pop("failed_at", None)
                task.pop("error", None)
                completed += 1
            else:
                task["status"] = "failed"
                task["failed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                task["error"] = "评分失败，详见控制台日志"
                failed += 1

        except Exception as e:
            print(f"❌ {scheme_id} 加载或校验失败：{type(e).__name__}: {e}")
            task["status"] = "failed"
            task["failed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            task["error"] = f"{type(e).__name__}: {e}"
            failed += 1

        update_next_task(task_queue)
        save_task_queue(task_queue)
        time.sleep(API_INTERVAL_SECONDS)

    print(f"\n{'=' * 60}")
    print("📊 评分完成总结")
    print("=" * 60)
    print(f"成功: {completed} 个")
    print(f"失败: {failed} 个")
    print(f"总计: {completed + failed} 个")
    print(f"\n结果保存在: {RESULTS_DIR}")
    print(f"任务队列已更新: {TASK_QUEUE_FILE}")


if __name__ == "__main__":
    main()
