import fitz  # PyMuPDF
import base64
from docx import Document

def pdf_to_base64_str(pdf_path: str) -> str:
    """
    PDF 파일을 읽어 base64 인코딩된 문자열로 반환합니다.

    Parameters
    ----------
    pdf_path : str
        PDF 파일 경로

    Returns
    -------
    str
        base64 인코딩된 PDF 문자열
    """
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    return pdf_b64

def txt_to_text_str(txt_path: str) -> str:
    """
    TXT 파일을 읽어 문자열로 반환합니다.

    Parameters
    ----------
    txt_path : str
        TXT 파일 경로

    Returns
    -------
    str
        텍스트 문자열
    """
    with open(txt_path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


def docx_to_text_str(docx_path: str) -> str:
    """
    DOCX 파일을 읽어 문자열로 반환합니다.

    Parameters
    ----------
    docx_path : str
        DOCX 파일 경로

    Returns
    -------
    str
        텍스트 문자열
    """


    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return "\n".join(full_text)