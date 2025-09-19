"""
Integrated Singleton-Aware Stage3 Node
MCL의 singleton 감지 능력을 보존한 최종 stage3 노드
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import os
import sys

# Add path for imports
sys.path.append('/home/cyyoon/test_area/ai_text_classification/2.langgraph')

from nodes.stage3_classification.singleton_aware_clustering import SingletonAwareClustering

def load_column_wise_data(data_dir: str) -> Dict[str, Dict[str, Tuple[pd.DataFrame, np.ndarray]]]:
    """Stage2 CSV 파일에서 컬럼별 데이터 로드"""
    
    result = {}
    
    # CSV 파일들 찾기
    import glob
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        
        # stage2_문4_img_20250918_112537.csv에서 문항 추출
        if file_name.startswith('stage2_'):
            parts = file_name.split('_')
            if len(parts) >= 2:
                question_id = parts[1]  # 문4, 문5 등
                
                print(f"📂 Loading {question_id} from {file_name}")
                
                try:
                    # CSV 파일 로드
                    df = pd.read_csv(csv_file)
                    print(f"   Loaded {len(df)} rows, columns: {list(df.columns)}")
                    
                    # 임베딩 컬럼들 찾기 (embed_text_1, embed_text_2 등)
                    embed_columns = [col for col in df.columns if col.startswith('embed_text_')]
                    text_columns = [col for col in df.columns if col.startswith('text_') and not col.startswith('embed_')]
                    
                    if question_id not in result:
                        result[question_id] = {}
                    
                    # 각 text/embedding 쌍별로 처리
                    for i, embed_col in enumerate(embed_columns):
                        # text_1 -> embed_text_1 매칭
                        text_col = f"text_{i+1}"
                        
                        if text_col in df.columns:
                            # 유효한 데이터만 필터링 (비어있지 않은 텍스트)
                            valid_mask = df[text_col].notna() & (df[text_col] != "")
                            valid_df = df[valid_mask].copy()
                            
                            if len(valid_df) > 0:
                                # 임베딩 데이터 파싱 (문자열에서 리스트로)
                                embeddings = []
                                for embed_str in valid_df[embed_col]:
                                    try:
                                        if isinstance(embed_str, str):
                                            # "[1.2, 3.4, ...]" 형식을 파싱
                                            embed_list = eval(embed_str)
                                            embeddings.append(embed_list)
                                        else:
                                            # 이미 리스트인 경우
                                            embeddings.append(embed_str)
                                    except:
                                        print(f"     Warning: Failed to parse embedding for {text_col}")
                                        continue
                                
                                if embeddings:
                                    embeddings_array = np.array(embeddings)
                                    column_name = text_col  # text_1, text_2 등
                                    
                                    result[question_id][column_name] = (valid_df, embeddings_array)
                                    print(f"   {column_name}: {len(valid_df)} rows, {embeddings_array.shape[1]} dims")
                        
                except Exception as e:
                    print(f"   ❌ Error loading {file_name}: {e}")
                    continue
    
    return result

def process_single_column_singleton_aware(
    column_name: str, 
    df: pd.DataFrame, 
    embeddings: np.ndarray
) -> Dict[str, Any]:
    """단일 컬럼에 대한 singleton-aware 클러스터링"""
    
    if len(embeddings) < 2:
        return {
            'column': column_name,
            'n_samples': len(embeddings),
            'algorithm': 'insufficient_data',
            'n_clusters': 1,
            'singleton_count': 0,
            'cluster_labels': [0] * len(embeddings),
            'silhouette_score': 0.0,
            'quality_score': 0.0
        }
    
    print(f"   🎯 Processing {column_name} ({len(embeddings)} embeddings)")
    
    # Singleton-aware clustering 실행
    clustering = SingletonAwareClustering()
    labels, info = clustering.fit_predict(embeddings)
    
    # 결과 정리
    result = {
        'column': column_name,
        'n_samples': len(embeddings),
        'algorithm': info['algorithm'],
        'n_clusters': info['n_clusters'],
        'singleton_count': info['singleton_count'],
        'cluster_labels': labels.tolist(),
        'silhouette_score': info['silhouette'],
        'quality_score': info['score']
    }
    
    print(f"     ✅ {info['algorithm']}: {info['n_clusters']} clusters, {info['singleton_count']} singletons")
    print(f"        Silhouette: {info['silhouette']:.3f}, Score: {info['score']:.3f}")
    
    return result

def process_question_singleton_aware(
    question_id: str, 
    question_data: Dict[str, Tuple[pd.DataFrame, np.ndarray]]
) -> Dict[str, Any]:
    """문항별 singleton-aware 클러스터링 처리"""
    
    print(f"\n📋 Processing {question_id}")
    print(f"=" * 40)
    
    results = {
        'question_id': question_id,
        'n_columns': len(question_data),
        'column_results': [],
        'total_clusters': 0,
        'total_singletons': 0,
        'algorithms_used': []
    }
    
    # 컬럼별 처리
    for column_name, (df, embeddings) in question_data.items():
        column_result = process_single_column_singleton_aware(column_name, df, embeddings)
        results['column_results'].append(column_result)
        results['total_clusters'] += column_result['n_clusters']
        results['total_singletons'] += column_result['singleton_count']
        
        if column_result['algorithm'] not in results['algorithms_used']:
            results['algorithms_used'].append(column_result['algorithm'])
    
    # 요약 통계
    avg_silhouette = np.mean([r['silhouette_score'] for r in results['column_results']])
    avg_quality = np.mean([r['quality_score'] for r in results['column_results']])
    
    results['avg_silhouette'] = avg_silhouette
    results['avg_quality_score'] = avg_quality
    
    print(f"\n📊 {question_id} Summary:")
    print(f"   Columns processed: {results['n_columns']}")
    print(f"   Total clusters: {results['total_clusters']}")
    print(f"   Total singletons: {results['total_singletons']}")
    print(f"   Algorithms used: {results['algorithms_used']}")
    print(f"   Average silhouette: {avg_silhouette:.3f}")
    print(f"   Average quality: {avg_quality:.3f}")
    
    return results

def singleton_aware_stage3_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Singleton-aware stage3 classification node"""
    
    print("\n🎯 Stage3: Singleton-Aware Classification")
    print("=" * 50)
    
    # 입력 데이터 경로 설정
    output_dir = state.get('output_dir', '/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/stage2_output')
    
    if not os.path.exists(output_dir):
        print(f"❌ Output directory not found: {output_dir}")
        return {**state, 'stage3_results': {'error': 'Output directory not found'}}
    
    try:
        # 1. 컬럼별 데이터 로드
        print("📂 Loading column-wise data...")
        data = load_column_wise_data(output_dir)
        
        if not data:
            print("❌ No data found")
            return {**state, 'stage3_results': {'error': 'No data found'}}
        
        # 2. 문항별 singleton-aware 클러스터링
        all_results = {
            'processing_type': 'singleton_aware_clustering',
            'questions': {},
            'overall_summary': {}
        }
        
        total_clusters = 0
        total_singletons = 0
        all_algorithms = []
        all_silhouettes = []
        all_quality_scores = []
        
        for question_id, question_data in data.items():
            question_result = process_question_singleton_aware(question_id, question_data)
            all_results['questions'][question_id] = question_result
            
            # 전체 통계 누적
            total_clusters += question_result['total_clusters']
            total_singletons += question_result['total_singletons']
            all_algorithms.extend(question_result['algorithms_used'])
            all_silhouettes.append(question_result['avg_silhouette'])
            all_quality_scores.append(question_result['avg_quality_score'])
        
        # 3. 전체 요약
        unique_algorithms = list(set(all_algorithms))
        overall_avg_silhouette = np.mean(all_silhouettes)
        overall_avg_quality = np.mean(all_quality_scores)
        
        all_results['overall_summary'] = {
            'total_questions': len(data),
            'total_clusters': total_clusters,
            'total_singletons': total_singletons,
            'singleton_ratio': total_singletons / (total_clusters + total_singletons) if (total_clusters + total_singletons) > 0 else 0,
            'algorithms_used': unique_algorithms,
            'overall_avg_silhouette': overall_avg_silhouette,
            'overall_avg_quality': overall_avg_quality
        }
        
        # 4. 결과 출력
        print(f"\n🎉 Singleton-Aware Stage3 Complete!")
        print(f"=" * 40)
        print(f"Questions processed: {len(data)}")
        print(f"Total clusters: {total_clusters}")
        print(f"Total singletons: {total_singletons}")
        print(f"Singleton ratio: {all_results['overall_summary']['singleton_ratio']:.3f}")
        print(f"Algorithms used: {unique_algorithms}")
        print(f"Overall silhouette: {overall_avg_silhouette:.3f}")
        print(f"Overall quality: {overall_avg_quality:.3f}")
        
        # 5. MCL vs Current 비교
        print(f"\n📊 Singleton Detection Analysis:")
        if total_singletons > 0:
            print(f"   ✅ Singleton detection working: {total_singletons} singletons found")
            print(f"   ✅ Singleton ratio: {all_results['overall_summary']['singleton_ratio']:.1%}")
            print(f"   ✅ Natural clustering: {total_clusters} meaningful clusters")
        else:
            print(f"   ⚠️  No singletons detected - check data characteristics")
        
        return {**state, 'stage3_results': all_results}
        
    except Exception as e:
        print(f"❌ Error in singleton-aware stage3: {e}")
        import traceback
        traceback.print_exc()
        return {**state, 'stage3_results': {'error': str(e)}}

# 테스트 함수
def test_singleton_aware_stage3():
    """Singleton-aware stage3 노드 테스트"""
    
    print("🧪 Testing Singleton-Aware Stage3 Node")
    print("=" * 50)
    
    # 테스트 state
    test_state = {
        'output_dir': '/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/temp_data/stage2_results'
    }
    
    # 노드 실행
    result_state = singleton_aware_stage3_node(test_state)
    
    # 결과 확인
    if 'error' in result_state.get('stage3_results', {}):
        print(f"❌ Test failed: {result_state['stage3_results']['error']}")
    else:
        print(f"✅ Test completed successfully!")
        
        # 상세 결과 출력
        results = result_state['stage3_results']
        summary = results['overall_summary']
        
        print(f"\n📋 Test Results Summary:")
        print(f"   Questions: {summary['total_questions']}")
        print(f"   Clusters: {summary['total_clusters']}")
        print(f"   Singletons: {summary['total_singletons']}")
        print(f"   Singleton ratio: {summary['singleton_ratio']:.1%}")
        print(f"   Quality score: {summary['overall_avg_quality']:.3f}")
    
    return result_state

if __name__ == "__main__":
    print("🎯 Singleton-Aware Stage3 Node")
    print("=" * 40)
    
    # 테스트 실행
    test_result = test_singleton_aware_stage3()
    
    print(f"\n✅ Singleton-aware stage3 node ready!")
    print(f"Key improvements:")
    print(f"• MCL singleton detection preserved")
    print(f"• Automatic outlier identification")  
    print(f"• Quality-based algorithm selection")
    print(f"• Reduced cluster proliferation")