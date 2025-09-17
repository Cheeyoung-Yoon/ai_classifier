from __future__ import annotations

import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Iterable, Tuple, Dict, Any, Optional, List
import os
import re
from pathlib import Path


@dataclass
class SmartExcelOptions:
    """스마트 로더 동작 옵션(간소화 버전)."""
    sheet_names: Optional[Iterable[str]] = None        # None이면 모든 시트
    skiprows_range: Iterable[int] = range(0, 6)        # 시도할 헤더 오프셋
    index_col_candidates: Tuple[str, ...] = ("IDKEY", "ID", "UID")
    require_index: bool = False                        # True면 인덱스 없는 후보는 후보군에 넣지 않음
    engine: Optional[str] = None
    object_threshold: float = 0.8                      # object-like 컬럼 판정 임계치


@dataclass
class SmartExcelResult:
    """스마트 로더 결과."""
    dataframe_path: str  # DataFrame이 저장된 CSV 파일 경로
    meta: Dict[str, Any] = field(default_factory=dict)
    
    # dataframe property를 제거하여 메모리 효율성 향상
    # 필요시 utils.data_utils.DataHelper.load_dataframe() 사용
    
    def as_dict(self) -> Dict[str, Any]:
        result = self.meta.copy()
        result['dataframe_path'] = self.dataframe_path
        # dataframe은 더 이상 자동으로 포함하지 않음
        return result


class SmartExcelLoader:
    """
    후보를 입력 순서대로 적재하고, 마지막(valid) 후보를 선택하는 간소화 로더.
    - 시트 순서: 엑셀의 sheet 순서 혹은 options.sheet_names 순서
    - skiprows 순서: options.skiprows_range 순서
    - require_index=True면 인덱스 후보만 후보군에 포함
    """

    def __init__(self, options: Optional[SmartExcelOptions] = None):
        self.options = options or SmartExcelOptions()
    
    @staticmethod
    def _extract_project_name(file_path: str) -> str:
        """파일 경로에서 프로젝트 이름 추출"""
        # 파일명에서 프로젝트 이름 추출 (확장자 제거)
        filename = Path(file_path).stem
        
        # 특수문자나 숫자로 시작하는 부분 제거하고 의미있는 이름 추출
        # 예: "-SUV_776부.xlsx" -> "SUV"
        # 예: "-2025년 하반기 NBCI_편의점_20250523_만족도점수_QC코딩완료.xlsx" -> "NBCI"
        
        # 앞의 특수문자 제거
        clean_name = re.sub(r'^[-_\d년월일\s]+', '', filename)
        
        # 첫 번째 의미있는 단어 추출
        match = re.search(r'([A-Za-z가-힣]+)', clean_name)
        if match:
            project_name = match.group(1)
        else:
            project_name = "default"
        
        return project_name.lower()
    
    @staticmethod
    def _create_data_dir_and_save(df: pd.DataFrame, file_path: str) -> str:
        """DataFrame을 프로젝트별 디렉토리에 저장"""
        project_name = SmartExcelLoader._extract_project_name(file_path)
        
        # 현재 작업 디렉토리 기준으로 데이터 디렉토리 생성
        current_dir = os.getcwd()
        data_dir = Path(current_dir) / "data" / project_name
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # CSV 파일로 저장
        csv_path = data_dir / "raw_data.csv"
        df.to_csv(csv_path, encoding='utf-8-sig', index=True)
        
        return str(csv_path)

    # ---------------------------
    # Public API
    # ---------------------------
    def excel_load(self, raw_path: str) -> SmartExcelResult:
        opt = self.options
        xls = pd.ExcelFile(raw_path, engine=opt.engine)
        all_sheets = list(opt.sheet_names) if opt.sheet_names else list(xls.sheet_names)

        candidates: List[Dict[str, Any]] = []

        for sheet in all_sheets:
            for sr in opt.skiprows_range:
                try:
                    df = pd.read_excel(raw_path, sheet_name=sheet, skiprows=sr, engine=opt.engine)
                    df = self._normalize_columns(df).dropna(how="all")
                    if df is None or df.empty:
                        continue

                    idx_col = self._find_index_col(df.columns, opt.index_col_candidates)

                    # ✅ 인덱스 없으면 무조건 스킵
                    if not idx_col:
                        continue

                    # 인덱스 적용
                    df_out = df.dropna(subset=[idx_col]).set_index(idx_col)

                    candidates.append({
                        "sheet": sheet,
                        "skiprows": sr,
                        "index_col": idx_col,
                        "n_rows": len(df_out),
                        "n_cols": df_out.shape[1],
                        "_df": df_out,
                    })
                except Exception:
                    continue

        if not candidates:
            raise ValueError("IDKEY/UID/ID 컬럼을 가진 유효한 시트를 찾지 못했습니다.")

        # 입력 순서대로 쌓였으므로 마지막 걸 선택
        chosen = candidates[-1]
        df_out = chosen["_df"].copy()

        object_columns = self._find_objectish_columns(df_out, threshold=opt.object_threshold)
        
        # object_columns에 해당하는 컬럼의 label만 추출
        column_labels = []
        if chosen["skiprows"] > 0:
            labels_df = pd.read_excel(
                raw_path,
                sheet_name=chosen["sheet"],
                nrows=chosen["skiprows"],
                engine=opt.engine
            )
            
            # 라벨 DataFrame에서 Q컬럼을 찾아서 해당하는 라벨을 매핑
            # labels_df에서 값이 object_columns에 있는 컬럼을 찾음
            for label_col in labels_df.columns:
                values = labels_df[label_col].dropna().astype(str).tolist()
                # 각 값이 object_columns에 있는지 확인
                for value in values:
                    if value in object_columns and value.strip() and value != 'nan':
                        # [실제컬럼명, 라벨명] 형태로 저장
                        column_labels.append([value, label_col])
        
        # object_columns에 해당하지만 label이 없는 컬럼들은 컬럼명만 포함
        labeled_columns = {label_list[0] for label_list in column_labels}
        for col in object_columns:
            if col not in labeled_columns:
                column_labels.append([col])  # 컬럼명만 포함

        meta = {
            "sheet_name": chosen["sheet"],
            "skiprows": chosen["skiprows"],
            "index_col": chosen["index_col"],
            "object_columns": object_columns,
            # "candidates": [{k: v for k, v in c.items() if k != "_df"} for c in candidates],
            # "options": asdict(opt),
            "column_labels": column_labels,
        }
        
        # DataFrame을 CSV 파일로 저장하고 경로 반환
        saved_path = self._create_data_dir_and_save(df_out, raw_path)
        
        return SmartExcelResult(dataframe_path=saved_path, meta=meta)

    # ---------------------------
    # Static / private helpers
    # ---------------------------
    @staticmethod
    def _flatten_columns(cols) -> List[str]:
        if isinstance(cols, pd.MultiIndex):
            return ["_".join([str(c) for c in tup if str(c) != "nan"]).strip() for tup in cols]
        return [str(c).strip() for c in cols]

    @classmethod
    def _normalize_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = cls._flatten_columns(df.columns)
        df.columns = [c.replace("Unnamed: ", "").strip() for c in df.columns]
        return df

    @staticmethod
    def _find_index_col(cols: Iterable[str],
                        candidates: Tuple[str, ...] = ("IDKEY", "ID", "UID")) -> Optional[str]:
        lower_map = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        # 흔한 변형 케이스 몇 가지 추가 매칭(옵션)
        alt = ("id_key", "idkey", "key", "번호", "식별자")
        for cand in alt:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        return None

    @staticmethod
    def _find_objectish_columns(df: pd.DataFrame, threshold: float = 0.8) -> List[str]:
        """
        문자열/범주 컬럼 탐지:
        - dtype 기반(string/object/category)
        - 값의 대부분이 문자열(str)인 경우 threshold 이상이면 포함
        """
        dfi = df.convert_dtypes()
        obj_like = dfi.select_dtypes(include=["string", "object", "category"]).columns.tolist()

        def is_mostly_str(series: pd.Series) -> bool:
            non_null = series.dropna()
            if len(non_null) == 0:
                return False
            str_ratio = non_null.apply(lambda x: isinstance(x, str)).mean()
            return float(str_ratio) >= threshold

        filtered = [c for c in obj_like if is_mostly_str(df[c])]
        drop_name = ['START_TIME','END_TIME','DATA_TIME','ACCESS_KEY',
                     'USER_AGENT','USER_DEVICE','IS_MOBILE','IP','RESULT',
                     'LAST_QUESTION','LAST_BEFORE_QUESTION','면접원']
        return [c for c in filtered if c not in drop_name]
