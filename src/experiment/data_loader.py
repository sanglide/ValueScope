"""
Data loading module
Responsible for loading value scenarios, value models, and human annotation data
"""

import csv
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValueDefinition:
    """Value definition data class"""
    layer: str  # L2 or L3
    value_id: str
    value_name: str
    schwartz_mapping: str
    definition: str
    notes: str = ""


@dataclass
class ValueScenarioSample:
    """Value scenario sample data class"""
    sample_id: str
    scenario_content: str  # Code or scenario description
    ground_truth_has_risk: bool  # Human annotation: whether risk exists
    ground_truth_values: list[str] = field(default_factory=list)  # Human-annotated value ID list
    metadata: dict = field(default_factory=dict)  # Additional metadata


class ValueModelLoader:
    """Value model loader"""
    
    def __init__(self, tables_dir: str):
        self.tables_dir = Path(tables_dir)
        self.l2_values: dict[str, ValueDefinition] = {}
        self.l3_values: dict[str, ValueDefinition] = {}
    
    def load(self) -> None:
        """Load all value models"""
        self._load_l2_values()
        self._load_l3_values()
    
    def _load_l2_values(self) -> None:
        """Load L2-layer value themes"""
        l2_file = self.tables_dir / "L2_Value_Themes.csv"
        if not l2_file.exists():
            print(f"Warning: L2 value file does not exist: {l2_file}")
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
        """Load L3-layer system value themes"""
        l3_file = self.tables_dir / "L3_system_value_themes.csv"
        if not l3_file.exists():
            print(f"Warning: L3 value file does not exist: {l3_file}")
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
        """Get all value definitions"""
        return {**self.l2_values, **self.l3_values}
    
    def format_value_model_for_prompt(self, include_l2: bool = True, include_l3: bool = True) -> str:
        """Format value model for prompt construction"""
        lines = []
        
        if include_l2 and self.l2_values:
            lines.append("### Human Value Themes (L2)")
            for vid, vdef in self.l2_values.items():
                lines.append(f"- **{vid} ({vdef.value_name})**: {vdef.definition}")
        
        if include_l3 and self.l3_values:
            lines.append("\n### System Value Themes (L3)")
            for vid, vdef in self.l3_values.items():
                lines.append(f"- **{vid} ({vdef.value_name})**: {vdef.definition}")
        
        return "\n".join(lines)


class ScenarioDataLoader:
    """Value scenario data loader"""
    
    def __init__(self, data_file: Optional[str] = None):
        self.data_file = Path(data_file) if data_file else None
        self.samples: list[ValueScenarioSample] = []
    
    def load_from_json(self, json_file: str) -> None:
        """Load scenario data from a JSON file"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            sample = ValueScenarioSample(
                sample_id=item.get('id', ''),
                scenario_content=item.get('scenario', ''),
                ground_truth_has_risk=item.get('has_value_risk', False),
                ground_truth_values=item.get('ground_truth_values', []),
                metadata=item.get('metadata', {})
            )
            self.samples.append(sample)
    
    def load_from_csv(self, csv_file: str) -> None:
        """Load scenario data from a CSV file"""
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse ground_truth_values (assumed to be comma-separated)
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
    
    def add_sample(self, sample: ValueScenarioSample) -> None:
        """Manually add a sample"""
        self.samples.append(sample)
    
    def get_samples(self) -> list[ValueScenarioSample]:
        """Get all samples"""
        return self.samples


def create_sample_dataset() -> list[ValueScenarioSample]:
    """Create a sample dataset (for testing purposes)"""
    samples = [
        ValueScenarioSample(
            sample_id="sample_001",
            scenario_content="""
def collect_user_data(user):
    # Collect all browsing history of the user
    history = get_browser_history(user.device_id)
    # Collect location information
    locations = get_location_history(user.device_id)
    # Upload to server
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
    # Determine loan eligibility based on zip code
    if applicant.zip_code in HIGH_RISK_AREAS:
        return {"eligible": False, "reason": "Location-based restriction"}
    # Determine eligibility based on age
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
    """Save the sample dataset to a JSON file"""
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
    print(f"Sample dataset saved to: {output_file}")


# ==============================================================================
# Issues Dataset Loader (values-issues-dataset-master)
# ==============================================================================

# Mapping: proposed_values_id in issues dataset -> value_id in this project
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
    """Loader for the values-issues-dataset-master dataset
    
    Aggregation granularity is at the issue level: concatenates all post texts
    under the same issue, and merges (with deduplication) all post-level value
    annotations as ground truth.
    """
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
    
    def load(
        self,
        sample_per_project: Optional[int] = None,
        max_text_length: int = 8000,
        seed: int = 42
    ) -> list[ValueScenarioSample]:
        """Load and return issue-level scenario samples
        
        Args:
            sample_per_project: Number of samples per project (None for all)
            max_text_length: Maximum text length in characters (truncates overly long issues)
            seed: Random seed
        """
        # 1. Load raw data
        issues = self._load_issues()
        posts = self._load_posts()
        labels = self._load_labels()
        
        # 2. Aggregate post texts by issue
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
        
        # 3. Aggregate value annotations by issue (deduplicated, excluding None)
        issue_values = {}
        for label in labels:
            iid = label["issue_id"]
            vid = label["proposed_values_id"]
            if iid not in issue_values:
                issue_values[iid] = set()
            mapped = ISSUES_VALUE_ID_MAPPING.get(vid)
            if mapped:  # Exclude None mappings
                issue_values[iid].add(mapped)
        
        # 4. Build samples
        samples_by_project: dict[str, list[ValueScenarioSample]] = {}
        
        for issue in issues:
            iid = issue["issue_id"]
            project = issue.get("project_name", "unknown")
            title = issue.get("title", "")
            
            # Concatenate texts
            texts_data = issue_texts.get(iid, {"title_text": title, "post_texts": []})
            full_text = f"[Issue Title]: {texts_data['title_text']}\n\n"
            for i, pt in enumerate(texts_data["post_texts"], 1):
                full_text += f"[Post {i}]:\n{pt}\n\n"
            
            # Truncate overly long texts
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
        
        # 5. Sampling
        if sample_per_project is not None:
            rng = random.Random(seed)
            sampled = []
            for project, project_samples in sorted(samples_by_project.items()):
                # Balanced sampling: maintain the ratio of risk/no-risk as much as possible
                with_risk = [s for s in project_samples if s.ground_truth_has_risk]
                without_risk = [s for s in project_samples if not s.ground_truth_has_risk]
                
                n_risk = min(len(with_risk), sample_per_project // 2 + sample_per_project % 2)
                n_no_risk = min(len(without_risk), sample_per_project - n_risk)
                # If one category is insufficient, compensate from the other
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
        """Load issues.csv"""
        filepath = self.dataset_dir / "issues.csv"
        with open(filepath, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f, delimiter='|'))
    
    def _load_posts(self) -> list[dict]:
        """Load issue-posts.csv"""
        filepath = self.dataset_dir / "issue-posts.csv"
        with open(filepath, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f, delimiter='|'))
    
    def _load_labels(self) -> list[dict]:
        """Load values-label.csv"""
        filepath = self.dataset_dir / "values-label.csv"
        with open(filepath, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f, delimiter='|'))
    
    def get_statistics(self) -> dict:
        """Get dataset statistics"""
        issues = self._load_issues()
        labels = self._load_labels()
        posts = self._load_posts()
        
        # Project distribution
        projects = {}
        for i in issues:
            p = i["project_name"]
            projects[p] = projects.get(p, 0) + 1
        
        # Issue-level aggregated annotations
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
        
        # Value distribution
        value_counts = {}
        for vids in issue_values.values():
            for vid in vids:
                value_counts[vid] = value_counts.get(vid, 0) + 1
        
        # Post text lengths
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
