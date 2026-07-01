"""
数据加载模块
负责加载价值场景、价值模型和人工标注数据
"""

import csv
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValueDefinition:
    """价值定义数据类"""
    layer: str  # L2 或 L3
    value_id: str
    value_name: str
    schwartz_mapping: str
    definition: str
    notes: str = ""


@dataclass
class ValueScenarioSample:
    """价值场景样本数据类"""
    sample_id: str
    scenario_content: str  # 代码或场景描述
    ground_truth_has_risk: bool  # 人工标注：是否有风险
    ground_truth_values: list[str] = field(default_factory=list)  # 人工标注的价值ID列表
    metadata: dict = field(default_factory=dict)  # 其他元数据


class ValueModelLoader:
    """价值模型加载器"""
    
    def __init__(self, tables_dir: str):
        self.tables_dir = Path(tables_dir)
        self.l2_values: dict[str, ValueDefinition] = {}
        self.l3_values: dict[str, ValueDefinition] = {}
    
    def load(self) -> None:
        """加载所有价值模型"""
        self._load_l2_values()
        self._load_l3_values()
    
    def _load_l2_values(self) -> None:
        """加载L2层价值主题"""
        l2_file = self.tables_dir / "L2_Value_Themes.csv"
        if not l2_file.exists():
            print(f"警告: L2价值文件不存在: {l2_file}")
            return
        
        with open(l2_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                value_def = ValueDefinition(
                    layer=row.get('layer', 'L2'),
                    value_id=row.get('value_id', ''),
                    value_name=row.get('value_name', ''),
                    schwartz_mapping=row.get('schwartz_mapping', ''),
                    definition=row.get('paper_definition', ''),
                    notes=row.get('notes', '')
                )
                self.l2_values[value_def.value_id] = value_def
    
    def _load_l3_values(self) -> None:
        """加载L3层系统价值主题"""
        l3_file = self.tables_dir / "L3_system_value_themes.csv"
        if not l3_file.exists():
            print(f"警告: L3价值文件不存在: {l3_file}")
            return
        
        with open(l3_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                value_def = ValueDefinition(
                    layer=row.get('layer', 'L3'),
                    value_id=row.get('value_id', ''),
                    value_name=row.get('value_name', ''),
                    schwartz_mapping=row.get('schwartz_mapping', ''),
                    definition=row.get('paper_definition', ''),
                    notes=row.get('mapping_notes', '')
                )
                self.l3_values[value_def.value_id] = value_def
    
    def get_all_values(self) -> dict[str, ValueDefinition]:
        """获取所有价值定义"""
        return {**self.l2_values, **self.l3_values}
    
    def format_value_model_for_prompt(self, include_l2: bool = True, include_l3: bool = True) -> str:
        """格式化价值模型用于prompt"""
        lines = []
        
        if include_l2 and self.l2_values:
            lines.append("### 人类价值主题 (L2)")
            for vid, vdef in self.l2_values.items():
                lines.append(f"- **{vid} ({vdef.value_name})**: {vdef.definition}")
        
        if include_l3 and self.l3_values:
            lines.append("\n### 系统价值主题 (L3)")
            for vid, vdef in self.l3_values.items():
                lines.append(f"- **{vid} ({vdef.value_name})**: {vdef.definition}")
        
        return "\n".join(lines)


class ScenarioDataLoader:
    """价值场景数据加载器"""
    
    def __init__(self, data_file: Optional[str] = None):
        self.data_file = Path(data_file) if data_file else None
        self.samples: list[ValueScenarioSample] = []
    
    def load_from_json(self, json_file: str) -> None:
        """从JSON文件加载场景数据"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        existing_ids = {s.sample_id for s in self.samples}
        for item in data:
            sample_id = item.get('id', '')
            if sample_id in existing_ids:
                continue
            sample = ValueScenarioSample(
                sample_id=sample_id,
                scenario_content=item.get('scenario', ''),
                ground_truth_has_risk=item.get('has_value_risk', False),
                ground_truth_values=item.get('ground_truth_values', []),
                metadata=item.get('metadata', {})
            )
            self.samples.append(sample)
            existing_ids.add(sample_id)
    
    def load_from_csv(self, csv_file: str) -> None:
        """从CSV文件加载场景数据"""
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 解析ground_truth_values（假设用逗号分隔）
                gt_values = []
                if row.get('ground_truth_values'):
                    gt_values = [v.strip() for v in row['ground_truth_values'].split(',')]
                
                sample = ValueScenarioSample(
                    sample_id=row.get('id', ''),
                    scenario_content=row.get('scenario', ''),
                    ground_truth_has_risk=row.get('has_value_risk', '').lower() in ('true', '1', 'yes'),
                    ground_truth_values=gt_values,
                    metadata={}
                )
                self.samples.append(sample)

    def load_from_directory(self, directory: str, pattern: str = "*_scenarios.json") -> int:
        """从目录中加载所有匹配的 JSON 文件
        
        Args:
            directory: 目录路径
            pattern: 文件名 glob 模式 (默认 *_scenarios.json)
        
        Returns:
            加载的样本总数
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory}")
        
        count = 0
        for json_file in sorted(dir_path.glob(pattern)):
            before = len(self.samples)
            self.load_from_json(str(json_file))
            loaded = len(self.samples) - before
            if loaded > 0:
                print(f"  [glob] {json_file.name}: {loaded} samples")
                count += loaded
        
        return count
    
    def add_sample(self, sample: ValueScenarioSample) -> None:
        """手动添加样本"""
        self.samples.append(sample)
    
    def get_samples(self) -> list[ValueScenarioSample]:
        """获取所有样本"""
        return self.samples


def create_sample_dataset() -> list[ValueScenarioSample]:
    """创建示例数据集（用于测试）"""
    samples = [
        ValueScenarioSample(
            sample_id="sample_001",
            scenario_content="""
def collect_user_data(user):
    # 收集用户的所有浏览历史
    history = get_browser_history(user.device_id)
    # 收集位置信息
    locations = get_location_history(user.device_id)
    # 上传到服务器
    upload_to_server({
        'user_id': user.id,
        'browser_history': history,
        'locations': locations,
        'device_info': get_device_info()
    })
""",
            ground_truth_has_risk=True,
            ground_truth_values=["HV9", "HV10"],  # Privacy, Security
            metadata={"source": "synthetic", "category": "data_collection"}
        ),
        ValueScenarioSample(
            sample_id="sample_002",
            scenario_content="""
def calculate_loan_eligibility(applicant):
    # 基于邮政编码判断贷款资格
    if applicant.zip_code in HIGH_RISK_AREAS:
        return {"eligible": False, "reason": "Location-based restriction"}
    # 基于年龄判断
    if applicant.age > 60:
        return {"eligible": False, "reason": "Age restriction"}
    return {"eligible": True}
""",
            ground_truth_has_risk=True,
            ground_truth_values=["HV3", "HV4"],  # Dignity, Inclusiveness
            metadata={"source": "synthetic", "category": "discrimination"}
        ),
        ValueScenarioSample(
            sample_id="sample_003",
            scenario_content="""
def display_user_profile(user_id):
    user = get_user(user_id)
    return {
        "name": user.name,
        "avatar": user.avatar_url,
        "public_bio": user.bio
    }
""",
            ground_truth_has_risk=False,
            ground_truth_values=[],
            metadata={"source": "synthetic", "category": "normal"}
        ),
    ]
    return samples


def save_sample_dataset(output_file: str) -> None:
    """保存示例数据集到JSON文件"""
    samples = create_sample_dataset()
    data = []
    for sample in samples:
        data.append({
            "id": sample.sample_id,
            "scenario": sample.scenario_content,
            "has_value_risk": sample.ground_truth_has_risk,
            "ground_truth_values": sample.ground_truth_values,
            "metadata": sample.metadata
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"示例数据集已保存到: {output_file}")


# ==============================================================================
# Issues Dataset Loader (values_issues_dataset)
# ==============================================================================

# 映射: issues数据集的proposed_values_id -> 本项目的value_id
ISSUES_VALUE_ID_MAPPING = {
    "1": None,       # None -> no value
    "2": "HV9",      # Privacy
    "3": "HV6",      # Freedom
    "4": "HV7",      # Independence
    "5": "SV6",      # Efficiency
    "6": "HV10",     # Security
    "7": "SV10",     # Longevity
    "10": "SV8",     # Usability
    "11": "SV9",     # Accessibility
    "12": "HV2",     # Pleasure
    "13": "SV2",     # Correctness
    "14": "HV5",     # Sense of belonging
    "15": "HV1",     # Conformity
    "16": "SV1",     # Trust
    "17": "HV3",     # Dignity
    "18": "HV8",     # Wealth
    "19": "HV4",     # Inclusiveness
    "20": "SV5",     # Reliability
    "21": "SV3",     # Compatibility
    "22": "SV4",     # Portability
    "23": "SV7",     # Energy Preservation
}


class IssuesDatasetLoader:
    """加载values_issues_dataset数据集（CSV原始格式）

    聚合粒度为issue级别：将同一issue下所有post文本拼接，
    将所有post的价值标注合并去重作为ground truth。

    注意：主实验现在统一使用 text_scenarios/issues.json（已规范化格式）。
    本加载器保留用于从 text_scenarios/source/ 下的 CSV 原始文件重建数据。
    """
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
    
    def load(
        self,
        sample_per_project: Optional[int] = None,
        max_text_length: int = 8000,
        seed: int = 42
    ) -> list[ValueScenarioSample]:
        """加载并返回issue级别的场景样本
        
        Args:
            sample_per_project: 每个项目采样数量（None表示全量）
            max_text_length: 文本最大字符数（截断过长的issue）
            seed: 随机种子
        """
        # 1. 加载原始数据
        issues = self._load_issues()
        posts = self._load_posts()
        labels = self._load_labels()
        
        # 2. 按issue聚合post文本
        issue_texts = {}
        for post in posts:
            iid = post["issue_id"]
            body = post.get("body_text", "")
            ptype = post.get("type", "post")
            if iid not in issue_texts:
                issue_texts[iid] = {"title_text": "", "post_texts": []}
            if ptype == "title":
                issue_texts[iid]["title_text"] = body
            else:
                issue_texts[iid]["post_texts"].append(body)
        
        # 3. 按issue聚合value标注（去重，排除None）
        issue_values = {}
        for label in labels:
            iid = label["issue_id"]
            vid = label["proposed_values_id"]
            if iid not in issue_values:
                issue_values[iid] = set()
            mapped = ISSUES_VALUE_ID_MAPPING.get(vid)
            if mapped:  # 排除None映射
                issue_values[iid].add(mapped)
        
        # 4. 构建样本
        samples_by_project: dict[str, list[ValueScenarioSample]] = {}
        
        for issue in issues:
            iid = issue["issue_id"]
            project = issue.get("project_name", "unknown")
            title = issue.get("title", "")
            
            # 拼接文本
            texts_data = issue_texts.get(iid, {"title_text": title, "post_texts": []})
            full_text = f"[Issue Title]: {texts_data['title_text']}\n\n"
            for i, pt in enumerate(texts_data["post_texts"], 1):
                full_text += f"[Post {i}]:\n{pt}\n\n"
            
            # 截断过长文本
            if len(full_text) > max_text_length:
                full_text = full_text[:max_text_length] + "\n\n[... truncated ...]"
            
            # ground truth
            gt_values = sorted(issue_values.get(iid, set()))
            has_risk = len(gt_values) > 0
            
            sample = ValueScenarioSample(
                sample_id=f"issue_{iid}",
                scenario_content=full_text,
                ground_truth_has_risk=has_risk,
                ground_truth_values=gt_values,
                metadata={
                    "source": "values-issues-dataset",
                    "scenario_type": "text",
                    "project_name": project,
                    "issue_title": title,
                    "artefact_url": issue.get("artefact_url", ""),
                    "text_length": len(full_text),
                }
            )
            
            if project not in samples_by_project:
                samples_by_project[project] = []
            samples_by_project[project].append(sample)
        
        # 5. 采样
        if sample_per_project is not None:
            rng = random.Random(seed)
            sampled = []
            for project, project_samples in sorted(samples_by_project.items()):
                # 平衡采样：尽量保持有风险/无风险的比例
                with_risk = [s for s in project_samples if s.ground_truth_has_risk]
                without_risk = [s for s in project_samples if not s.ground_truth_has_risk]
                
                n_risk = min(len(with_risk), sample_per_project // 2 + sample_per_project % 2)
                n_no_risk = min(len(without_risk), sample_per_project - n_risk)
                # 如果某类不够，从另一类补
                if n_no_risk < sample_per_project - n_risk:
                    n_risk = min(len(with_risk), sample_per_project - n_no_risk)
                
                sampled.extend(rng.sample(with_risk, min(n_risk, len(with_risk))))
                sampled.extend(rng.sample(without_risk, min(n_no_risk, len(without_risk))))
            return sampled
        else:
            all_samples = []
            for project_samples in samples_by_project.values():
                all_samples.extend(project_samples)
            return all_samples
    
    def _load_issues(self) -> list[dict]:
        """加载issues.csv"""
        filepath = self.dataset_dir / "source" / "issues.csv"
        if not filepath.exists():
            filepath = self.dataset_dir / "issues.csv"
        with open(filepath, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f, delimiter='|'))
    
    def _load_posts(self) -> list[dict]:
        """加载issue-posts.csv"""
        filepath = self.dataset_dir / "source" / "issue-posts.csv"
        if not filepath.exists():
            filepath = self.dataset_dir / "issue-posts.csv"
        with open(filepath, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f, delimiter='|'))
    
    def _load_labels(self) -> list[dict]:
        """加载values-label.csv"""
        filepath = self.dataset_dir / "source" / "values-label.csv"
        if not filepath.exists():
            filepath = self.dataset_dir / "values-label.csv"
        with open(filepath, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f, delimiter='|'))
    
    def get_statistics(self) -> dict:
        """获取数据集统计信息"""
        issues = self._load_issues()
        labels = self._load_labels()
        posts = self._load_posts()
        
        # 项目分布
        projects = {}
        for i in issues:
            p = i["project_name"]
            projects[p] = projects.get(p, 0) + 1
        
        # issue级别聚合标注
        issue_values = {}
        for l in labels:
            iid = l["issue_id"]
            vid = l["proposed_values_id"]
            if iid not in issue_values:
                issue_values[iid] = set()
            mapped = ISSUES_VALUE_ID_MAPPING.get(vid)
            if mapped:
                issue_values[iid].add(mapped)
        
        issues_with_risk = sum(1 for v in issue_values.values() if v)
        issues_no_risk = len(issues) - issues_with_risk
        
        # 价值分布
        value_counts = {}
        for vids in issue_values.values():
            for vid in vids:
                value_counts[vid] = value_counts.get(vid, 0) + 1
        
        # post文本长度
        issue_texts_len = {}
        for p in posts:
            iid = p["issue_id"]
            issue_texts_len[iid] = issue_texts_len.get(iid, 0) + len(p.get("body_text", ""))
        
        lengths = list(issue_texts_len.values())
        lengths.sort()
        
        return {
            "total_issues": len(issues),
            "total_posts": len(posts),
            "total_labels": len(labels),
            "project_distribution": projects,
            "issues_with_value_risk": issues_with_risk,
            "issues_no_value_risk": issues_no_risk,
            "value_distribution": dict(sorted(value_counts.items(), key=lambda x: -x[1])),
            "text_length_stats": {
                "min": lengths[0] if lengths else 0,
                "max": lengths[-1] if lengths else 0,
                "mean": sum(lengths) / len(lengths) if lengths else 0,
                "median": lengths[len(lengths) // 2] if lengths else 0,
            }
        }
