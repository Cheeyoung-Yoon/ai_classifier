# utils/cost_tracker.py - LLM Cost Tracking and Stage Management Utilities
from typing import List, Dict, Any, Optional
from datetime import datetime

def calculate_pipeline_runtime(start_time: str, end_time: Optional[str] = None) -> float:
    """
    Calculate pipeline runtime in seconds
    
    Args:
        start_time: Start time in ISO format
        end_time: End time in ISO format (current time if None)
        
    Returns:
        Runtime in seconds
    """
    try:
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time) if end_time else datetime.now()
        runtime = (end_dt - start_dt).total_seconds()
        return round(runtime, 2)
    except Exception:
        return 0.0

def format_runtime_display(runtime_seconds: float) -> str:
    """
    Format runtime for human-readable display
    
    Args:
        runtime_seconds: Runtime in seconds
        
    Returns:
        Formatted runtime string
    """
    if runtime_seconds < 60:
        return f"{runtime_seconds:.1f}s"
    elif runtime_seconds < 3600:
        minutes = int(runtime_seconds // 60)
        seconds = runtime_seconds % 60
        return f"{minutes}m {seconds:.1f}s"
    else:
        hours = int(runtime_seconds // 3600)
        minutes = int((runtime_seconds % 3600) // 60)
        seconds = runtime_seconds % 60
        return f"{hours}h {minutes}m {seconds:.1f}s"

def calculate_total_llm_cost(llm_logs: Optional[List[Dict[str, Any]]]) -> float:
    """
    Calculate total LLM usage cost from logs
    
    Args:
        llm_logs: List of LLM usage logs containing cost information
        
    Returns:
        Total cost in USD (rounded to 6 decimal places)
    """
    if not llm_logs:
        return 0.0
    
    total_cost = 0.0
    for log in llm_logs:
        if isinstance(log, dict) and 'cost_usd' in log:
            cost = log['cost_usd']
            if isinstance(cost, (int, float)):
                total_cost += cost
    
    return round(total_cost, 6)

def calculate_total_tokens(llm_logs: Optional[List[Dict[str, Any]]]) -> Dict[str, int]:
    """
    Calculate total token usage from logs
    
    Args:
        llm_logs: List of LLM usage logs
        
    Returns:
        Dictionary with prompt_tokens, completion_tokens, and total_tokens
    """
    if not llm_logs:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    prompt_tokens = 0
    completion_tokens = 0
    
    for log in llm_logs:
        if isinstance(log, dict):
            prompt_tokens += log.get('prompt_tokens', 0)
            completion_tokens += log.get('completion_tokens', 0)
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens
    }

def get_llm_usage_summary(llm_logs: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Get comprehensive LLM usage summary
    
    Args:
        llm_logs: List of LLM usage logs
        
    Returns:
        Dictionary with cost, token, and call statistics
    """
    if not llm_logs:
        return {
            "total_calls": 0,
            "total_cost_usd": 0.0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "models_used": []
        }
    
    total_cost = calculate_total_llm_cost(llm_logs)
    token_stats = calculate_total_tokens(llm_logs)
    models_used = list(set(log.get('model', 'unknown') for log in llm_logs if isinstance(log, dict)))
    
    return {
        "total_calls": len(llm_logs),
        "total_cost_usd": total_cost,
        "total_tokens": token_stats["total_tokens"],
        "prompt_tokens": token_stats["prompt_tokens"],
        "completion_tokens": token_stats["completion_tokens"],
        "models_used": models_used
    }

def create_stage_info(stage_name: str, llm_logs: Optional[List[Dict[str, Any]]] = None, 
                     start_time: Optional[str] = None, completion_time: Optional[str] = None,
                     pipeline_start_time: Optional[str] = None) -> Dict[str, Any]:
    """
    Create stage information dictionary with runtime tracking
    
    Args:
        stage_name: Name of the stage
        llm_logs: LLM usage logs for this stage
        start_time: Stage start time (ISO format)
        completion_time: Stage completion time (ISO format)
        pipeline_start_time: Pipeline start time for cumulative runtime calculation
        
    Returns:
        Stage information dictionary
    """
    llm_summary = get_llm_usage_summary(llm_logs)
    current_time = completion_time or datetime.now().isoformat()
    
    stage_info = {
        "stage_name": stage_name,
        "start_time": start_time,
        "completion_time": current_time,
        "llm_usage": llm_summary,
        "status": "completed" if completion_time else "in_progress"
    }
    
    # Add runtime calculations if pipeline_start_time is provided
    if pipeline_start_time:
        cumulative_runtime = calculate_pipeline_runtime(pipeline_start_time, current_time)
        stage_info["cumulative_runtime_seconds"] = cumulative_runtime
        stage_info["cumulative_runtime_display"] = format_runtime_display(cumulative_runtime)
        
        if start_time:
            stage_runtime = calculate_pipeline_runtime(start_time, current_time)
            stage_info["stage_runtime_seconds"] = stage_runtime
            stage_info["stage_runtime_display"] = format_runtime_display(stage_runtime)
    
    return stage_info

def format_cost_summary(cost_usd: float) -> str:
    """
    Format cost for display
    
    Args:
        cost_usd: Cost in USD
        
    Returns:
        Formatted cost string
    """
    if cost_usd == 0:
        return "$0.00"
    elif cost_usd < 0.01:
        return f"${cost_usd:.6f}"
    else:
        return f"${cost_usd:.4f}"

def print_stage_summary(stage_name: str, llm_logs: Optional[List[Dict[str, Any]]] = None,
                       pipeline_start_time: Optional[str] = None, stage_runtime: Optional[float] = None):
    """
    Print formatted stage summary to console with runtime information
    
    Args:
        stage_name: Name of the stage
        llm_logs: LLM usage logs
        pipeline_start_time: Pipeline start time for cumulative runtime
        stage_runtime: Stage-specific runtime in seconds
    """
    llm_summary = get_llm_usage_summary(llm_logs)
    current_time = datetime.now()
    
    print(f"\n{'='*60}")
    print(f"STAGE COMPLETED: {stage_name}")
    print(f"{'='*60}")
    
    # Runtime information
    if pipeline_start_time:
        cumulative_runtime = calculate_pipeline_runtime(pipeline_start_time)
        print(f"Cumulative Runtime: {format_runtime_display(cumulative_runtime)}")
        
    if stage_runtime:
        print(f"Stage Runtime: {format_runtime_display(stage_runtime)}")
    
    # LLM usage information
    print(f"Total LLM Calls: {llm_summary['total_calls']}")
    print(f"Total Tokens: {llm_summary['total_tokens']:,}")
    print(f"  - Prompt Tokens: {llm_summary['prompt_tokens']:,}")
    print(f"  - Completion Tokens: {llm_summary['completion_tokens']:,}")
    print(f"Total Cost: {format_cost_summary(llm_summary['total_cost_usd'])}")
    if llm_summary['models_used']:
        print(f"Models Used: {', '.join(llm_summary['models_used'])}")
    
    print(f"Completion Time: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

def get_pipeline_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get comprehensive pipeline summary with runtime and cost information
    
    Args:
        state: Current graph state
        
    Returns:
        Dictionary with pipeline summary information
    """
    pipeline_start = state.get('pipeline_start_time')
    current_stage = state.get('current_stage', 'UNKNOWN')
    total_cost = state.get('total_llm_cost_usd', 0.0)
    llm_logs = state.get('llm_logs', [])
    stage_history = state.get('stage_history', [])
    
    summary = {
        "current_stage": current_stage,
        "total_stages_completed": len(stage_history),
        "pipeline_start_time": pipeline_start,
        "current_time": datetime.now().isoformat(),
        "total_cost_usd": total_cost,
        "llm_usage": get_llm_usage_summary(llm_logs)
    }
    
    if pipeline_start:
        runtime = calculate_pipeline_runtime(pipeline_start)
        summary["total_runtime_seconds"] = runtime
        summary["total_runtime_display"] = format_runtime_display(runtime)
    
    return summary

def print_pipeline_status(state: Dict[str, Any]):
    """
    Print current pipeline status with runtime and cost information
    
    Args:
        state: Current graph state
    """
    summary = get_pipeline_summary(state)
    
    print(f"\n🔄 PIPELINE STATUS")
    print(f"{'─'*40}")
    print(f"Current Stage: {summary['current_stage']}")
    print(f"Stages Completed: {summary['total_stages_completed']}")
    
    if 'total_runtime_display' in summary:
        print(f"Total Runtime: {summary['total_runtime_display']}")
    
    print(f"Total Cost: {format_cost_summary(summary['total_cost_usd'])}")
    print(f"Total LLM Calls: {summary['llm_usage']['total_calls']}")
    print(f"Total Tokens: {summary['llm_usage']['total_tokens']:,}")
    print(f"{'─'*40}\n")