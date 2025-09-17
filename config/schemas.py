"""
Pydantic schemas for structured LLM outputs extracted from prompt.config.yaml
All schemas match the exact JSON formats defined in the prompt configurations
"""
from typing import Dict, List, Optional, Literal, Union
from pydantic import BaseModel, Field, RootModel


# ========== Survey Parser Schema ==========
class SurveyQuestionItem(BaseModel):
    """Individual question item from survey parser"""
    question_that_is_related: Optional[str] = Field(description="Related question ID or null")
    question_text: str = Field(description="Full question text")
    open_question_number: str = Field(description="Original ID or integer")
    question_type: Literal["img", "concept", "depend", "pos_neg", "depend_pos_neg", "etc", "etc_pos_neg"] = Field(
        description="Question type classification"
    )


class SurveyParserSchema(RootModel[List[SurveyQuestionItem]]):
    """Survey parser output - array of questions"""
    pass


# ========== Question Data Matcher Schema ==========
class QuestionDataMatcherSchema(RootModel[Dict[str, List[str]]]):
    """Question to column mapping - dynamic keys with column lists"""
    pass


# ========== Grammar Check Schema ==========
class GrammarCorrectionSchema(BaseModel):
    """Grammar correction result"""
    corrected: str = Field(description="Corrected answer text")


# ========== Question Analysis Schema ==========
class QuestionAnalysisSchema(BaseModel):
    """Question choices analysis"""
    question: str = Field(description="Question summary")
    answers: Dict[str, str] = Field(description="Answer choices with explanations")


# ========== Sentence Analysis Schema ==========
class SVCKeywords(BaseModel):
    """S/V/C keywords for a single sentence"""
    S: List[str] = Field(default_factory=list, description="Subject keywords")
    V: List[str] = Field(default_factory=list, description="Verb keywords in -다 form") 
    C: List[str] = Field(default_factory=list, description="Complement/object keywords")


class SentenceAnalysisSchema(BaseModel):
    """Sentence analysis result for all sentence types"""
    matching_question: bool = Field(description="Whether answer matches question")
    pos_neg: Literal["POSITIVE", "NEGATIVE", "NEUTRAL"] = Field(description="Sentiment classification")
    automic_sentence: List[str] = Field(
        default_factory=list,
        max_items=3,
        description="Atomic sentences (max 3)"
    )
    SVC_keywords: Dict[str, SVCKeywords] = Field(
        default_factory=dict,
        description="SVC keywords for each sentence"
    )


# ========== Schema Registry ==========
SCHEMA_REGISTRY = {
    # Survey processing
    "survey_parser_4.1": SurveyParserSchema,
    "question_data_matcher": QuestionDataMatcherSchema,
    "survey_context_summarizer": str,  # Simple string output
    
    # Grammar and question analysis  
    "sentence_grammar_check": GrammarCorrectionSchema,
    "sentence_question_analysis": QuestionAnalysisSchema,
    
    # Sentence analysis (all variants use same schema)
    "sentence_depend_pos_neg_split": SentenceAnalysisSchema,
    "sentence_pos_neg_split": SentenceAnalysisSchema,
    "sentence_depend_split": SentenceAnalysisSchema,
    "sentence_only": SentenceAnalysisSchema,
}