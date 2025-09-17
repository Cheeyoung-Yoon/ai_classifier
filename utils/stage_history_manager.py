# utils/stage_history_manager.py - Independent Stage History Management
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

class StageHistoryManager:
    """
    Stage history를 별도 파일로 관리하는 매니저 클래스
    State와 독립적으로 pipeline의 전체 진행 과정을 추적
    """
    
    def __init__(self, project_name: str, pipeline_id: Optional[str] = None, 
                 history_dir: str = "pipeline_history"):
        """
        Initialize stage history manager
        
        Args:
            project_name: 프로젝트 이름
            pipeline_id: 파이프라인 고유 ID (없으면 자동 생성)
            history_dir: History 파일들을 저장할 디렉토리
        """
        self.project_name = project_name
        self.pipeline_id = pipeline_id or self._generate_pipeline_id()
        self.history_dir = Path(history_dir)
        self.history_dir.mkdir(parents=True, exist_ok=True)
        
        # History 파일 경로
        self.history_file = self.history_dir / f"{self.pipeline_id}_history.json"
        
        # History 데이터 구조
        self.history_data = {
            "pipeline_id": self.pipeline_id,
            "project_name": self.project_name,
            "pipeline_start_time": datetime.now().isoformat(),
            "last_updated": None,
            "stages": [],
            "metadata": {
                "total_stages": 0,
                "total_runtime_seconds": 0.0,
                "total_llm_cost_usd": 0.0,
                "total_llm_calls": 0,
                "total_tokens": 0
            }
        }
        
        # 초기 파일 생성
        self._save_history()
    
    def _generate_pipeline_id(self) -> str:
        """파이프라인 고유 ID 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return f"{self.project_name}_{timestamp}"
    
    def add_stage(self, stage_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        새 stage 정보를 history에 추가
        
        Args:
            stage_info: Stage 완료 정보
            
        Returns:
            업데이트된 metadata
        """
        # Stage 정보에 순서 번호 추가
        stage_info["stage_number"] = len(self.history_data["stages"]) + 1
        stage_info["added_at"] = datetime.now().isoformat()
        
        # History에 추가
        self.history_data["stages"].append(stage_info)
        self.history_data["last_updated"] = datetime.now().isoformat()
        
        # 메타데이터 업데이트
        self._update_metadata()
        
        # 파일 저장
        self._save_history()
        
        return self.history_data["metadata"].copy()
    
    def _update_metadata(self):
        """메타데이터 업데이트"""
        stages = self.history_data["stages"]
        
        if not stages:
            return
        
        # 기본 통계
        self.history_data["metadata"]["total_stages"] = len(stages)
        
        # 런타임 계산
        last_stage = stages[-1]
        self.history_data["metadata"]["total_runtime_seconds"] = last_stage.get(
            "cumulative_runtime_seconds", 0.0
        )
        
        # LLM 사용량 합계 (누적이 아닌 각 stage의 실제 사용량)
        total_cost = 0.0
        total_calls = 0
        total_tokens = 0
        
        for stage in stages:
            llm_usage = stage.get("llm_usage", {})
            # 각 stage에서 실제 증가한 양만 계산
            stage_cost = llm_usage.get("stage_cost_increment", 0.0)
            stage_calls = llm_usage.get("stage_calls_increment", 0)
            stage_tokens = llm_usage.get("stage_tokens_increment", 0)
            
            total_cost += stage_cost
            total_calls += stage_calls  
            total_tokens += stage_tokens
        
        self.history_data["metadata"]["total_llm_cost_usd"] = total_cost
        self.history_data["metadata"]["total_llm_calls"] = total_calls
        self.history_data["metadata"]["total_tokens"] = total_tokens
    
    def _save_history(self):
        """History를 파일에 저장"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.history_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save stage history: {e}")
    
    def get_history(self) -> Dict[str, Any]:
        """전체 history 데이터 반환"""
        return self.history_data.copy()
    
    def get_metadata(self) -> Dict[str, Any]:
        """메타데이터만 반환"""
        return self.history_data["metadata"].copy()
    
    def get_stage_summary(self) -> List[Dict[str, Any]]:
        """Stage별 요약 정보 반환"""
        summary = []
        for stage in self.history_data["stages"]:
            summary.append({
                "stage_number": stage.get("stage_number", 0),
                "stage_name": stage.get("stage_name", "Unknown"),
                "runtime_seconds": stage.get("stage_runtime_seconds", 0.0),
                "cumulative_runtime": stage.get("cumulative_runtime_seconds", 0.0),
                "cost_usd": stage.get("llm_usage", {}).get("stage_cost_increment", 0.0),
                "completion_time": stage.get("completion_time", "Unknown")
            })
        return summary
    
    def print_current_status(self):
        """현재 상태 출력"""
        metadata = self.history_data["metadata"]
        last_stage = self.history_data["stages"][-1] if self.history_data["stages"] else None
        
        from utils.cost_tracker import format_runtime_display, format_cost_summary
        
        print(f"\n📊 PIPELINE HISTORY STATUS")
        print(f"{'─'*50}")
        print(f"Pipeline ID: {self.pipeline_id}")
        print(f"Project: {self.project_name}")
        print(f"Total Stages: {metadata['total_stages']}")
        
        if metadata['total_runtime_seconds'] > 0:
            print(f"Total Runtime: {format_runtime_display(metadata['total_runtime_seconds'])}")
        
        print(f"Total Cost: {format_cost_summary(metadata['total_llm_cost_usd'])}")
        print(f"Total LLM Calls: {metadata['total_llm_calls']}")
        print(f"Total Tokens: {metadata['total_tokens']:,}")
        
        if last_stage:
            print(f"Last Stage: {last_stage.get('stage_name', 'Unknown')}")
            print(f"Last Updated: {self.history_data['last_updated']}")
        
        print(f"History File: {self.history_file}")
        print(f"{'─'*50}\n")
    
    def get_history_file_path(self) -> str:
        """History 파일 경로 반환 (state에 저장용)"""
        return str(self.history_file)

# 전역 히스토리 매니저 인스턴스들을 관리하는 레지스트리
_history_managers: Dict[str, StageHistoryManager] = {}

def get_or_create_history_manager(project_name: str, pipeline_id: Optional[str] = None) -> StageHistoryManager:
    """
    프로젝트별 히스토리 매니저 획득 또는 생성
    
    Args:
        project_name: 프로젝트 이름
        pipeline_id: 파이프라인 ID (없으면 새로 생성)
    
    Returns:
        StageHistoryManager 인스턴스
    """
    key = pipeline_id or project_name
    
    if key not in _history_managers:
        _history_managers[key] = StageHistoryManager(project_name, pipeline_id)
    
    return _history_managers[key]

def cleanup_history_managers():
    """메모리의 히스토리 매니저들 정리"""
    global _history_managers
    _history_managers.clear()