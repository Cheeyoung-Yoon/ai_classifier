# tools/file_preprocess/file_loader.py
import os
from io_layer.file_loader.load_doc import pdf_to_base64_str, txt_to_text_str, docx_to_text_str
from io_layer.file_loader.smart_load_excel import SmartExcelLoader

class UnsupportedFileError(ValueError): ...

class FileLoader:
    def __init__(self):
        self.excel = SmartExcelLoader()

    def load_survey(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            result = pdf_to_base64_str(path)
        elif ext == ".txt":
            result = txt_to_text_str(path)
        elif ext == ".docx":
            result = docx_to_text_str(path)

        else:
            raise UnsupportedFileError(f"지원하지 않는 파일 형식: {ext}")

        return {"type": ext,
                "path" : path,
                "text" : result,
                "meta": {"size_bytes": os.path.getsize(path)}}
    
    
    def smart_load_excel(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        if ext in [".xls", ".xlsx" ]:
            result =  self.excel.excel_load(path)
        else:
            raise UnsupportedFileError(f"지원하지 않는 엑셀 파일 형식: {ext}")
        return {
                "path" : path,
                "dataframe_path": result.dataframe_path,
                "meta" : result.meta}