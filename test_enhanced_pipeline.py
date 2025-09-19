#!/usr/bin/env python3
"""실제 Stage2 데이터로 향상된 KNN→CSLS→MCL 파이프라인 전체 테스트"""

import pandas as pd
import numpy as np
import os
import sys
import time
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import markov_clustering as mc
import networkx as nx

# Add project root to path using config
try:
    from config.config import settings
    project_root = Path(settings.PROJECT_DATA_BASE_DIR).resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except ImportError:
    # Fallback for when config is not available
    project_root = Path(__file__).parent.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

def assess_quality(nmi_score, ari_score):
    """NMI와 ARI 점수를 기반으로 품질 등급 평가"""
    combined_score = (nmi_score + ari_score) / 2
    
    if combined_score >= 0.96:
        return "EXCELLENT", combined_score
    elif combined_score >= 0.88:
        return "GOOD", combined_score
    elif combined_score >= 0.83:
        return "FAIR", combined_score
    elif combined_score >= 0.72:
        return "IMPROVE", combined_score
    else:
        return "FAIL", combined_score

def try_knn_csls_mcl_pipeline(embeddings, texts, k=10, inflation=2.0):
    """향상된 KNN → CSLS → MCL 파이프라인"""
    
    n_samples = len(embeddings)
    if n_samples < 2:
        return list(range(n_samples)), "insufficient_data", 0.0, 0.0
    
    try:
        print(f"  🔄 KNN → CSLS → MCL 파이프라인 시작 (k={k}, inflation={inflation})")
        
        # 1. KNN 그래프 구성
        print(f"     1️⃣ KNN 그래프 구성 중...")
        nn = NearestNeighbors(n_neighbors=min(k, n_samples-1), metric='cosine')
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)
        
        # 2. CSLS (Cross-domain Similarity Local Scaling) 적용
        print(f"     2️⃣ CSLS 스케일링 적용 중...")
        # 각 포인트의 평균 유사도 계산
        cos_similarities = 1 - distances  # cosine distance를 similarity로 변환
        mean_similarities = np.mean(cos_similarities, axis=1)
        
        # CSLS 그래프 구성
        adjacency_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j_idx, j in enumerate(indices[i]):
                if i != j:
                    cos_ij = cos_similarities[i, j_idx]
                    # CSLS 공식: 2 * cos(xi, xj) - r(xi) - r(xj)
                    csls_score = 2 * cos_ij - mean_similarities[i] - mean_similarities[j]
                    
                    if csls_score > 0:  # 양수인 경우만 연결
                        adjacency_matrix[i, j] = csls_score
        
        # 대칭화
        adjacency_matrix = (adjacency_matrix + adjacency_matrix.T) / 2
        
        # 3. MCL 클러스터링
        print(f"     3️⃣ MCL 클러스터링 실행 중...")
        result = mc.run_mcl(adjacency_matrix, inflation=inflation)
        clusters = mc.get_clusters(result)
        
        if not clusters:
            return list(range(n_samples)), "mcl_failed", 0.0, 0.0
        
        # 클러스터 라벨 할당
        labels = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster_nodes in enumerate(clusters):
            for node in cluster_nodes:
                labels[node] = cluster_id
        
        print(f"     ✅ KNN→CSLS→MCL 완료: {len(clusters)}개 클러스터")
        
        return labels.tolist(), "knn_csls_mcl", len(clusters), adjacency_matrix.sum()
        
    except Exception as e:
        print(f"     ❌ KNN→CSLS→MCL 실패: {e}")
        return list(range(n_samples)), "knn_csls_mcl_failed", 0.0, 0.0

def match_labels_across_columns(results_by_column):
    """같은 질문 내 컬럼 간 라벨 매칭"""
    
    if len(results_by_column) < 2:
        return results_by_column
    
    print(f"  🔗 컬럼 간 라벨 매칭 시작 ({len(results_by_column)}개 컬럼)")
    
    try:
        # 각 컬럼의 클러스터 대표 텍스트 추출
        column_clusters = {}
        for column, result in results_by_column.items():
            if 'cluster_labels' not in result or 'texts' not in result:
                continue
            
            cluster_texts = {}
            for i, (text, label) in enumerate(zip(result['texts'], result['cluster_labels'])):
                if label not in cluster_texts:
                    cluster_texts[label] = []
                cluster_texts[label].append(text)
            
            # 각 클러스터의 대표 텍스트 (가장 긴 텍스트)
            cluster_representatives = {}
            for label, texts in cluster_texts.items():
                cluster_representatives[label] = max(texts, key=len)
            
            column_clusters[column] = cluster_representatives
        
        if len(column_clusters) < 2:
            return results_by_column
        
        # TF-IDF를 사용한 클러스터 간 유사도 계산
        all_representatives = []
        cluster_mapping = []
        
        for column, representatives in column_clusters.items():
            for label, text in representatives.items():
                all_representatives.append(text)
                cluster_mapping.append((column, label))
        
        if len(all_representatives) < 2:
            return results_by_column
        
        # TF-IDF 벡터화
        vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        tfidf_matrix = vectorizer.fit_transform(all_representatives)
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # 컬럼 간 클러스터 매칭
        base_column = list(column_clusters.keys())[0]
        matched_results = {base_column: results_by_column[base_column]}
        
        for target_column in list(column_clusters.keys())[1:]:
            if target_column not in results_by_column:
                continue
            
            base_indices = [i for i, (col, _) in enumerate(cluster_mapping) if col == base_column]
            target_indices = [i for i, (col, _) in enumerate(cluster_mapping) if col == target_column]
            
            # 최적 매칭 찾기
            label_mapping = {}
            for target_idx in target_indices:
                target_label = cluster_mapping[target_idx][1]
                best_similarity = -1
                best_base_label = 0
                
                for base_idx in base_indices:
                    base_label = cluster_mapping[base_idx][1]
                    sim = similarity_matrix[base_idx, target_idx]
                    
                    if sim > best_similarity:
                        best_similarity = sim
                        best_base_label = base_label
                
                label_mapping[target_label] = best_base_label
            
            # 라벨 재할당
            new_result = results_by_column[target_column].copy()
            if 'cluster_labels' in new_result:
                new_labels = [label_mapping.get(label, label) for label in new_result['cluster_labels']]
                new_result['cluster_labels'] = new_labels
                
                # 클러스터 수 업데이트
                new_result['n_clusters'] = len(set(new_labels))
            
            matched_results[target_column] = new_result
        
        print(f"     ✅ 컬럼 간 라벨 매칭 완료")
        return matched_results
        
    except Exception as e:
        print(f"     ❌ 라벨 매칭 실패: {e}")
        return results_by_column

def load_column_wise_data(directory_path):
    """질문별, 컬럼별로 데이터를 로드합니다."""
    data_groups = {}
    
    if not os.path.exists(directory_path):
        print(f"⚠️ Directory not found: {directory_path}")
        return data_groups
    
    csv_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    
    print(f"📁 Found {len(csv_files)} CSV files")
    
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            # 질문 ID 추출
            question_id = None
            if 'question_id' in df.columns and not df['question_id'].isna().all():
                question_id = df['question_id'].iloc[0]
            else:
                filename = os.path.basename(file_path)
                import re
                match = re.search(r'문(\d+)', filename)
                if match:
                    question_id = f"문{match.group(1)}"
            
            if not question_id:
                continue
            
            # 텍스트 컬럼 확인
            text_columns = []
            for col in df.columns:
                if col.startswith('text_') and 'embed' not in col:
                    text_columns.append(col)
            
            # 각 텍스트 컬럼에 대해 처리
            for text_col in text_columns:
                embed_col = f"embed_{text_col}"
                if embed_col not in df.columns:
                    continue
                
                valid_mask = df[text_col].notna() & df[embed_col].notna()
                valid_df = df[valid_mask].copy()
                
                if len(valid_df) == 0:
                    continue
                
                try:
                    embeddings = []
                    for embed_str in valid_df[embed_col]:
                        if isinstance(embed_str, str):
                            embed_array = np.array(eval(embed_str))
                        else:
                            embed_array = np.array(embed_str)
                        embeddings.append(embed_array)
                    
                    embeddings = np.array(embeddings)
                    group_key = f"{question_id}_{text_col}"
                    
                    data_groups[group_key] = {
                        'texts': valid_df[text_col].tolist(),
                        'embeddings': embeddings,
                        'question_id': question_id,
                        'column': text_col,
                        'file_path': file_path
                    }
                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            continue
    
    return data_groups

def test_enhanced_stage3_pipeline():
    """향상된 Stage3 파이프라인 전체 테스트"""
    
    print("🚀 향상된 Stage3 파이프라인 실제 데이터 테스트")
    print("=" * 70)
    
    # 실제 데이터 로딩
    data_directory = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/temp_data/stage2_results copy/"
    data_groups = load_column_wise_data(data_directory)
    
    if not data_groups:
        print("❌ 로딩된 데이터가 없습니다.")
        return
    
    print(f"📊 총 {len(data_groups)}개 데이터 그룹 로딩 완료")
    
    # 질문별로 그룹화
    questions = {}
    for group_key, data in data_groups.items():
        question_id = data['question_id']
        if question_id not in questions:
            questions[question_id] = {}
        questions[question_id][data['column']] = data
    
    print(f"📋 총 {len(questions)}개 질문 발견: {list(questions.keys())}")
    
    # 각 질문에 대해 처리
    all_results = {}
    
    for question_id, question_data in questions.items():
        print(f"\n{'=' * 50}")
        print(f"🔍 {question_id} 처리 중 ({len(question_data)}개 컬럼)")
        print(f"{'=' * 50}")
        
        start_time = time.time()
        
        # 각 컬럼에 대해 클러스터링 수행
        column_results = {}
        
        for column, data in question_data.items():
            print(f"\n📊 {column} 처리 중...")
            print(f"   • 샘플 수: {len(data['texts'])}")
            print(f"   • 임베딩 형태: {data['embeddings'].shape}")
            
            # KNN→CSLS→MCL 파이프라인 적용
            cluster_labels, algorithm, n_clusters, graph_weight = try_knn_csls_mcl_pipeline(
                data['embeddings'], 
                data['texts']
            )
            
            # 가상의 정답 라벨 생성 (실제로는 수동 라벨링된 데이터 사용)
            # 여기서는 텍스트 기반 간단한 라벨링 시뮬레이션
            true_labels = []
            unique_texts = list(set(data['texts']))
            text_to_label = {text: i for i, text in enumerate(unique_texts)}
            true_labels = [text_to_label[text] for text in data['texts']]
            
            # NMI, ARI 계산
            if len(set(cluster_labels)) > 1 and len(set(true_labels)) > 1:
                nmi_score = normalized_mutual_info_score(true_labels, cluster_labels)
                ari_score = adjusted_rand_score(true_labels, cluster_labels)
            else:
                nmi_score = 0.0
                ari_score = 0.0
            
            # 품질 평가
            quality_grade, combined_score = assess_quality(nmi_score, ari_score)
            
            column_results[column] = {
                'texts': data['texts'],
                'embeddings': data['embeddings'],
                'cluster_labels': cluster_labels,
                'true_labels': true_labels,
                'algorithm': algorithm,
                'n_clusters': n_clusters,
                'n_samples': len(data['texts']),
                'nmi_score': nmi_score,
                'ari_score': ari_score,
                'combined_score': combined_score,
                'quality_grade': quality_grade,
                'graph_weight': graph_weight
            }
            
            print(f"   ✅ {column} 완료:")
            print(f"      • 알고리즘: {algorithm}")
            print(f"      • 클러스터 수: {n_clusters}")
            print(f"      • NMI: {nmi_score:.4f}")
            print(f"      • ARI: {ari_score:.4f}")
            print(f"      • 품질: {quality_grade} ({combined_score:.4f})")
        
        # 컬럼 간 라벨 매칭
        if len(column_results) > 1:
            print(f"\n🔗 {question_id} 컬럼 간 라벨 매칭...")
            matched_results = match_labels_across_columns(column_results)
        else:
            matched_results = column_results
        
        # 결과 저장
        all_results[question_id] = matched_results
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️ {question_id} 처리 완료 ({elapsed_time:.2f}초)")
    
    # 전체 결과 요약
    print(f"\n{'=' * 70}")
    print("📊 전체 결과 요약")
    print(f"{'=' * 70}")
    
    total_samples = 0
    quality_counts = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "IMPROVE": 0, "FAIL": 0}
    algorithm_counts = {}
    
    for question_id, question_results in all_results.items():
        print(f"\n🔍 {question_id}:")
        
        for column, result in question_results.items():
            total_samples += result['n_samples']
            quality_counts[result['quality_grade']] += 1
            
            algorithm = result['algorithm']
            algorithm_counts[algorithm] = algorithm_counts.get(algorithm, 0) + 1
            
            print(f"   📊 {column}:")
            print(f"      • 샘플: {result['n_samples']}, 클러스터: {result['n_clusters']}")
            print(f"      • 알고리즘: {result['algorithm']}")
            print(f"      • NMI: {result['nmi_score']:.4f}, ARI: {result['ari_score']:.4f}")
            print(f"      • 품질: {result['quality_grade']} ({result['combined_score']:.4f})")
    
    print(f"\n📈 전체 통계:")
    print(f"   • 총 샘플 수: {total_samples:,}")
    print(f"   • 총 처리 그룹: {sum(len(qr) for qr in all_results.values())}")
    print(f"   • 품질 분포: {dict(quality_counts)}")
    print(f"   • 알고리즘 사용: {dict(algorithm_counts)}")
    
    print(f"\n✅ 향상된 Stage3 파이프라인 테스트 완료!")
    
    return all_results

if __name__ == "__main__":
    test_enhanced_stage3_pipeline()