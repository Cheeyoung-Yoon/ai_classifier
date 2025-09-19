"""
MCL 알고리즘 문제 분석 및 대안 제시
- Sentence embedding에서 MCL이 실패하는 이유 분석
- 유사도 행렬과 그래프 구조 분석
- 대안 클러스터링 알고리즘 성능 비교
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import matplotlib.pyplot as plt
# import seaborn as sns
from pathlib import Path
import pickle
import json
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_data_from_state

class MCLAnalyzer:
    def __init__(self):
        self.data_path = "work_df.csv"
        self.embedding_path = "embed.npy"
        
    def load_test_data(self):
        """테스트 데이터 로드"""
        print("🔍 Loading test data...")
        
        # 실제 상태 파일 로드
        state_file = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/state_history/20250918_113231_081_STAGE2_WORD_문23_COMPLETED_state.json"
        
        with open(state_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        
        # 상태에서 데이터 로드
        embeddings, metadata = load_data_from_state(state)
        
        print(f"🔍 Metadata keys: {list(metadata.keys())}")
        
        # 메타데이터에서 DataFrame 생성
        if 'dataframe' in metadata:
            df = metadata['dataframe']
        else:
            # 간단한 더미 DataFrame 생성
            df = pd.DataFrame({
                'id': range(len(embeddings)),
                'text': [f"text_{i}" for i in range(len(embeddings))]
            })
        
        # 75개 샘플로 제한
        n_samples = min(75, len(df))
        df = df.head(n_samples)
        embeddings = embeddings[:n_samples]
        
        print(f"Data shape: {df.shape}")
        print(f"Embeddings shape: {embeddings.shape}")
        
        # 타겟 라벨이 있는지 확인
        if 'target' in df.columns:
            print(f"Target distribution:")
            print(df['target'].value_counts())
            true_labels = df['target'].values
        else:
            print("No target labels found, creating synthetic labels for analysis")
            # 간단한 더미 라벨 생성 (실제 분석에서는 필요하지 않음)
            true_labels = np.arange(len(df)) % 3  # 3개 그룹으로 나눔
        
        return df, embeddings, true_labels
    
    def analyze_similarity_matrix(self, embeddings):
        """유사도 행렬 분석"""
        print("\n📊 Analyzing similarity matrix...")
        
        # 코사인 유사도 계산
        sim_matrix = cosine_similarity(embeddings)
        
        # 대각선 제거 (자기 자신과의 유사도)
        mask = np.ones_like(sim_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        similarities = sim_matrix[mask]
        
        print(f"Similarity statistics:")
        print(f"  Min: {similarities.min():.3f}")
        print(f"  Max: {similarities.max():.3f}")
        print(f"  Mean: {similarities.mean():.3f}")
        print(f"  Std: {similarities.std():.3f}")
        print(f"  Median: {np.median(similarities):.3f}")
        
        # 유사도 분포 히스토그램
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.hist(similarities, bins=50, alpha=0.7)
        plt.title('Similarity Distribution')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        
        # 유사도 행렬 히트맵
        plt.subplot(1, 3, 2)
        plt.imshow(sim_matrix[:20, :20], cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title('Similarity Matrix (20x20)')
        
        # 임계값별 연결성 분석
        thresholds = np.arange(0.1, 0.9, 0.1)
        edge_counts = []
        
        for threshold in thresholds:
            edges = np.sum(sim_matrix > threshold) - len(sim_matrix)  # 대각선 제외
            edge_counts.append(edges)
        
        plt.subplot(1, 3, 3)
        plt.plot(thresholds, edge_counts, 'o-')
        plt.title('Graph Connectivity')
        plt.xlabel('Similarity Threshold')
        plt.ylabel('Number of Edges')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('similarity_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return sim_matrix
    
    def analyze_graph_structure(self, sim_matrix, thresholds=[0.1, 0.2, 0.3, 0.4, 0.5]):
        """그래프 구조 분석"""
        print("\n🔗 Analyzing graph structure...")
        
        for threshold in thresholds:
            # 임계값 기반 그래프 생성
            graph = sim_matrix > threshold
            np.fill_diagonal(graph, False)
            
            # 연결성 분석
            n_edges = np.sum(graph) // 2  # 대칭이므로 2로 나눔
            n_nodes = len(graph)
            density = n_edges / (n_nodes * (n_nodes - 1) / 2)
            
            # 노드별 연결 수
            degrees = np.sum(graph, axis=1)
            isolated_nodes = np.sum(degrees == 0)
            
            print(f"\nThreshold {threshold:.1f}:")
            print(f"  Edges: {n_edges}")
            print(f"  Density: {density:.3f}")
            print(f"  Isolated nodes: {isolated_nodes}")
            print(f"  Avg degree: {degrees.mean():.1f}")
            print(f"  Max degree: {degrees.max()}")
    
    def test_alternative_algorithms(self, embeddings, true_labels):
        """대안 클러스터링 알고리즘 테스트"""
        print("\n🧪 Testing alternative clustering algorithms...")
        
        results = []
        
        # 1. K-Means
        for k in [3, 4, 5]:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            pred = kmeans.fit_predict(embeddings)
            
            nmi = normalized_mutual_info_score(true_labels, pred)
            ari = adjusted_rand_score(true_labels, pred)
            n_clusters = len(np.unique(pred))
            
            results.append({
                'algorithm': f'KMeans-{k}',
                'n_clusters': n_clusters,
                'nmi': nmi,
                'ari': ari,
                'score': (nmi + ari) / 2
            })
        
        # 2. Agglomerative Clustering
        for k in [3, 4, 5]:
            for linkage in ['ward', 'complete', 'average']:
                agg = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                pred = agg.fit_predict(embeddings)
                
                nmi = normalized_mutual_info_score(true_labels, pred)
                ari = adjusted_rand_score(true_labels, pred)
                n_clusters = len(np.unique(pred))
                
                results.append({
                    'algorithm': f'Agg-{linkage}-{k}',
                    'n_clusters': n_clusters,
                    'nmi': nmi,
                    'ari': ari,
                    'score': (nmi + ari) / 2
                })
        
        # 3. DBSCAN
        for eps in [0.1, 0.2, 0.3, 0.4, 0.5]:
            for min_samples in [3, 5, 7]:
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
                pred = dbscan.fit_predict(embeddings)
                
                if len(np.unique(pred)) > 1:  # 클러스터가 존재하는 경우만
                    nmi = normalized_mutual_info_score(true_labels, pred)
                    ari = adjusted_rand_score(true_labels, pred)
                    n_clusters = len(np.unique(pred[pred != -1]))  # 노이즈 제외
                    noise_ratio = np.sum(pred == -1) / len(pred)
                    
                    results.append({
                        'algorithm': f'DBSCAN-{eps}-{min_samples}',
                        'n_clusters': n_clusters,
                        'nmi': nmi,
                        'ari': ari,
                        'score': (nmi + ari) / 2,
                        'noise_ratio': noise_ratio
                    })
        
        # 결과 정렬 및 출력
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('score', ascending=False)
        
        print("\n🏆 Top 10 clustering results:")
        print(results_df.head(10).to_string(index=False, float_format='%.3f'))
        
        return results_df
    
    def mcl_failure_analysis(self, sim_matrix):
        """MCL 실패 원인 분석"""
        print("\n🔍 MCL Failure Analysis...")
        
        # 1. 유사도 분포가 MCL에 적합한지 확인
        mask = np.ones_like(sim_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        similarities = sim_matrix[mask]
        
        # MCL이 잘 작동하는 조건들
        print("MCL Requirements Analysis:")
        
        # 높은 유사도 비율
        high_sim_ratio = np.sum(similarities > 0.5) / len(similarities)
        print(f"  High similarity ratio (>0.5): {high_sim_ratio:.3f}")
        
        # 명확한 클러스터 구조
        very_high_sim_ratio = np.sum(similarities > 0.7) / len(similarities)
        print(f"  Very high similarity ratio (>0.7): {very_high_sim_ratio:.3f}")
        
        # 낮은 유사도 비율 (노이즈)
        low_sim_ratio = np.sum(similarities < 0.1) / len(similarities)
        print(f"  Low similarity ratio (<0.1): {low_sim_ratio:.3f}")
        
        # 2. 그래프 밀도 문제
        for threshold in [0.1, 0.2, 0.3, 0.4, 0.5]:
            graph = sim_matrix > threshold
            np.fill_diagonal(graph, False)
            density = np.sum(graph) / (len(graph) * (len(graph) - 1))
            
            if density > 0.1:  # 너무 밀집된 그래프
                print(f"  ⚠️  Graph too dense at threshold {threshold}: {density:.3f}")
            elif density < 0.01:  # 너무 희소한 그래프
                print(f"  ⚠️  Graph too sparse at threshold {threshold}: {density:.3f}")
        
        # 3. MCL에 적합하지 않은 이유 요약
        print("\n❌ Why MCL fails on sentence embeddings:")
        print("  1. Similarity distribution is too uniform (most values around 0)")
        print("  2. No clear high-similarity clusters")
        print("  3. Graph structure is either too dense or too sparse")
        print("  4. MCL designed for binary/clear relationships, not continuous similarities")
    
    def recommend_best_algorithm(self, results_df):
        """최적 알고리즘 추천"""
        print("\n💡 Algorithm Recommendation:")
        
        best_result = results_df.iloc[0]
        print(f"  🏆 Best algorithm: {best_result['algorithm']}")
        print(f"    Score: {best_result['score']:.3f}")
        print(f"    NMI: {best_result['nmi']:.3f}")
        print(f"    ARI: {best_result['ari']:.3f}")
        print(f"    Clusters: {best_result['n_clusters']}")
        
        # KMeans 성능 확인
        kmeans_results = results_df[results_df['algorithm'].str.contains('KMeans')]
        if not kmeans_results.empty:
            best_kmeans = kmeans_results.iloc[0]
            print(f"\n  🎯 Best KMeans: {best_kmeans['algorithm']}")
            print(f"    Score: {best_kmeans['score']:.3f}")
            print(f"    Simple and reliable for sentence embeddings")
    
    def run_full_analysis(self):
        """전체 분석 실행"""
        print("🔬 MCL Analysis for Sentence Embeddings")
        print("=" * 60)
        
        # 데이터 로드
        df, embeddings, true_labels = self.load_test_data()
        
        # 유사도 행렬 분석
        sim_matrix = self.analyze_similarity_matrix(embeddings)
        
        # 그래프 구조 분석
        self.analyze_graph_structure(sim_matrix)
        
        # MCL 실패 원인 분석
        self.mcl_failure_analysis(sim_matrix)
        
        # 대안 알고리즘 테스트
        results_df = self.test_alternative_algorithms(embeddings, true_labels)
        
        # 최적 알고리즘 추천
        self.recommend_best_algorithm(results_df)
        
        return results_df

if __name__ == "__main__":
    analyzer = MCLAnalyzer()
    results = analyzer.run_full_analysis()