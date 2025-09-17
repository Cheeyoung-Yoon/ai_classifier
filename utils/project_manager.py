"""
Project Directory Manager - 프로젝트별 디렉토리 구조 관리
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path


class ProjectDirectoryManager:
    """프로젝트별 디렉토리 구조를 관리하는 클래스"""
    
    def __init__(self, project_name: str, base_dir: str = "/home/cyyoon/test_area/ai_text_classification/2.langgraph"):
        self.project_name = project_name
        self.base_dir = base_dir
        self.project_dir = os.path.join(base_dir, "data", project_name)
        self.temp_data_dir = os.path.join(self.project_dir, "temp_data")
        self.state_file_path = os.path.join(self.project_dir, "state.json")
        
    def create_project_structure(self) -> Dict[str, str]:
        """프로젝트 디렉토리 구조 생성"""
        # 프로젝트 루트 디렉토리 생성
        os.makedirs(self.project_dir, exist_ok=True)
        
        # raw 데이터 디렉토리 생성 (외부 API에서 받아올 원본 데이터)
        raw_dir = os.path.join(self.project_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        # temp_data 디렉토리 생성
        os.makedirs(self.temp_data_dir, exist_ok=True)
        
        # Stage 2 결과 저장용 디렉토리
        stage2_dir = os.path.join(self.temp_data_dir, "stage2_results")
        os.makedirs(stage2_dir, exist_ok=True)
        
        # state_history 디렉토리 생성 (매 프로세스별 state 저장)
        state_history_dir = os.path.join(self.project_dir, "state_history")
        os.makedirs(state_history_dir, exist_ok=True)
        
        return {
            "project_dir": self.project_dir,
            "raw_dir": raw_dir,
            "temp_data_dir": self.temp_data_dir,
            "stage2_results_dir": stage2_dir,
            "state_history_dir": state_history_dir,
            "state_file": self.state_file_path
        }
    
    def get_stage2_csv_path(self, question_id: str, question_type: str, timestamp: str = None) -> str:
        """Stage 2 CSV 파일 경로 생성"""
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        stage2_dir = os.path.join(self.temp_data_dir, "stage2_results")
        os.makedirs(stage2_dir, exist_ok=True)
        
        return os.path.join(stage2_dir, f"stage2_{question_id}_{question_type}_{timestamp}.csv")
    
    def get_raw_file_path(self, filename: str) -> str:
        """Raw 데이터 파일 경로 반환"""
        raw_dir = os.path.join(self.project_dir, "raw")
        return os.path.join(raw_dir, filename)
    
    def copy_raw_files(self, source_survey_path: str, source_data_path: str) -> Dict[str, str]:
        """외부에서 받은 원본 파일들을 raw 디렉토리로 복사"""
        import shutil
        
        raw_dir = os.path.join(self.project_dir, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        
        # 파일명 추출
        survey_filename = os.path.basename(source_survey_path)
        data_filename = os.path.basename(source_data_path)
        
        # 대상 경로
        dest_survey_path = os.path.join(raw_dir, survey_filename)
        dest_data_path = os.path.join(raw_dir, data_filename)
        
        # 파일 복사
        if os.path.exists(source_survey_path):
            shutil.copy2(source_survey_path, dest_survey_path)
        if os.path.exists(source_data_path):
            shutil.copy2(source_data_path, dest_data_path)
        
        return {
            "survey_file_path": dest_survey_path,
            "data_file_path": dest_data_path
        }
    
    def save_state(self, state: Dict[str, Any], config: Dict[str, Any] = None) -> str:
        """상태를 state.json에 저장 및 state history 관리"""
        if config and config.get('save_state_log', True):
            timestamp = datetime.now()
            timestamp_str = timestamp.isoformat()
            
            # 현재 시간 추가
            state_with_timestamp = {
                **state,
                'saved_at': timestamp_str,
                'project_directory_structure': self.get_project_info()
            }
            
            # 최신 state.json 저장
            with open(self.state_file_path, 'w', encoding='utf-8') as f:
                json.dump(state_with_timestamp, f, ensure_ascii=False, indent=2, default=str)
            
            # state history 저장 (매 저장시마다)
            self.save_state_history(state_with_timestamp, timestamp)
            
            print(f"📄 State saved to: {self.state_file_path}")
            return self.state_file_path
        
        return None
    
    def save_state_history(self, state: Dict[str, Any], timestamp: datetime) -> str:
        """State history 저장"""
        state_history_dir = os.path.join(self.project_dir, "state_history")
        os.makedirs(state_history_dir, exist_ok=True)
        
        # 타임스탬프 기반 파일명
        timestamp_filename = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]  # 밀리초까지
        current_stage = state.get('current_stage', 'unknown')
        
        history_filename = f"{timestamp_filename}_{current_stage}_state.json"
        history_path = os.path.join(state_history_dir, history_filename)
        
        # State history 저장
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"📑 State history saved: {history_path}")
        return history_path
    
    def list_state_history(self) -> list:
        """State history 파일 목록 반환"""
        state_history_dir = os.path.join(self.project_dir, "state_history")
        if os.path.exists(state_history_dir):
            files = [f for f in os.listdir(state_history_dir) if f.endswith('_state.json')]
            return sorted(files)
        return []
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """저장된 상태 로드"""
        if os.path.exists(self.state_file_path):
            with open(self.state_file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return None
    
    def get_project_info(self) -> Dict[str, str]:
        """프로젝트 정보 반환"""
        return {
            "project_name": self.project_name,
            "project_dir": self.project_dir,
            "temp_data_dir": self.temp_data_dir,
            "state_file": self.state_file_path,
            "created_at": datetime.now().isoformat()
        }
    
    def clean_temp_data(self) -> bool:
        """임시 데이터 정리"""
        try:
            import shutil
            if os.path.exists(self.temp_data_dir):
                shutil.rmtree(self.temp_data_dir)
                os.makedirs(self.temp_data_dir, exist_ok=True)
                print(f"🧹 Cleaned temp data: {self.temp_data_dir}")
                return True
        except Exception as e:
            print(f"❌ Failed to clean temp data: {e}")
            return False
    
    def list_stage2_files(self) -> list:
        """Stage 2 결과 파일 목록"""
        stage2_dir = os.path.join(self.temp_data_dir, "stage2_results")
        if os.path.exists(stage2_dir):
            return [f for f in os.listdir(stage2_dir) if f.endswith('.csv')]
        return []


# 글로벌 매니저 인스턴스
_project_managers = {}

def get_project_manager(project_name: str, base_dir: str = None) -> ProjectDirectoryManager:
    """프로젝트 매니저 인스턴스 반환 (싱글톤 패턴)"""
    if project_name not in _project_managers:
        if base_dir:
            _project_managers[project_name] = ProjectDirectoryManager(project_name, base_dir)
        else:
            _project_managers[project_name] = ProjectDirectoryManager(project_name)
    return _project_managers[project_name]


def initialize_project_directories(project_name: str, base_dir: str = None) -> Dict[str, str]:
    """프로젝트 디렉토리 초기화"""
    manager = get_project_manager(project_name, base_dir)
    return manager.create_project_structure()