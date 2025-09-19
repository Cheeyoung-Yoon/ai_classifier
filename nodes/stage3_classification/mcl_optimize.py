"""
MCL 파라미터를 sentence embedding에 최적화
"""
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from mcl_pipeline import MCLPipeline
from evaluation import mcl_scoring_function

def optimize_mcl_for_sentences():
    """Sentence embedding에 최적화된 MCL 파라미터 찾기"""
    print("🔧 MCL Parameter Optimization for Sentence Embeddings")
    print("=" * 60)
    
    # 더 현실적인 sentence embedding 시뮬레이션
    np.random.seed(42)
    
    # 3개 그룹의 문장들 (더 현실적인 유사도)
    # 그룹 1: 연비 관련 문장들
    fuel_sentences = np.random.randn(25, 128) * 0.3 + np.array([0.8, 0.2] + [0]*126)
    
    # 그룹 2: 디자인 관련 문장들  
    design_sentences = np.random.randn(25, 128) * 0.3 + np.array([0.2, 0.8] + [0]*126)
    
    # 그룹 3: 성능 관련 문장들
    performance_sentences = np.random.randn(25, 128) * 0.3 + np.array([-0.5, -0.5] + [0]*126)
    
    embeddings = np.vstack([fuel_sentences, design_sentences, performance_sentences])
    true_labels = np.array([0]*25 + [1]*25 + [2]*25)
    
    print(f"Test data: {embeddings.shape[0]} sentence embeddings, 3 semantic groups")
    
    # 유사도 분석
    similarity_matrix = cosine_similarity(embeddings)
    print(f"Similarity range: {similarity_matrix.min():.3f} to {similarity_matrix.max():.3f}")
    print(f"Average similarity: {similarity_matrix.mean():.3f}")
    
    # sentence embedding에 맞는 파라미터 범위 테스트
    inflation_values = [1.1, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0]
    k_values = [3, 5, 8, 10, 15, 20]
    max_iter_values = [20, 50, 100]
    
    best_score = -1
    best_params = None
    best_details = None
    results = []
    
    print(f"\nTesting {len(inflation_values)} × {len(k_values)} × {len(max_iter_values)} = {len(inflation_values) * len(k_values) * len(max_iter_values)} combinations...")
    
    for inflation in inflation_values:
        for k in k_values:
            for max_iter in max_iter_values:
                if k >= len(embeddings):
                    continue
                    
                try:
                    start_time = time.time()
                    mcl = MCLPipeline(inflation=inflation, max_iters=max_iter)
                    mcl.fit(embeddings, k=k)
                    fit_time = time.time() - start_time
                    
                    summary = mcl.get_cluster_summary()
                    n_clusters = summary["n_clusters"]
                    
                    # 평가
                    scores = mcl_scoring_function(true_labels, mcl.cluster_labels)
                    nmi = scores['metrics']['nmi']
                    ari = scores['metrics']['ari']
                    composite_score = (nmi + ari) / 2
                    
                    singleton_ratio = scores['cluster_stats']['pred_singleton_ratio']
                    
                    result = {
                        'inflation': inflation,
                        'k': k,
                        'max_iter': max_iter,
                        'n_clusters': n_clusters,
                        'nmi': nmi,
                        'ari': ari,
                        'composite_score': composite_score,
                        'singleton_ratio': singleton_ratio,
                        'fit_time': fit_time
                    }
                    results.append(result)
                    
                    # 좋은 클러스터링 조건:
                    # 1. 적당한 클러스터 수 (2-10개)
                    # 2. 낮은 singleton 비율 (<0.3)
                    # 3. 높은 NMI/ARI 점수
                    if (2 <= n_clusters <= 10 and 
                        singleton_ratio < 0.3 and 
                        composite_score > best_score):
                        
                        best_score = composite_score
                        best_params = {
                            'inflation': inflation,
                            'k': k, 
                            'max_iter': max_iter
                        }
                        best_details = result
                        
                except Exception as e:
                    print(f"Failed: inflation={inflation}, k={k}, max_iter={max_iter} - {e}")
                    continue
    
    # 결과 분석
    print(f"\n📊 Results Analysis:")
    print(f"Total tested combinations: {len(results)}")
    
    if best_params:
        print(f"\n🏆 Best Parameters Found:")
        print(f"  Inflation: {best_params['inflation']}")
        print(f"  K neighbors: {best_params['k']}")
        print(f"  Max iterations: {best_params['max_iter']}")
        print(f"  Clusters found: {best_details['n_clusters']}")
        print(f"  NMI: {best_details['nmi']:.3f}")
        print(f"  ARI: {best_details['ari']:.3f}")
        print(f"  Composite score: {best_details['composite_score']:.3f}")
        print(f"  Singleton ratio: {best_details['singleton_ratio']:.3f}")
        print(f"  Fit time: {best_details['fit_time']:.3f}s")
    else:
        print("❌ No good parameters found!")
        
        # 차선책: 가장 높은 점수를 가진 결과들 보기
        if results:
            results.sort(key=lambda x: x['composite_score'], reverse=True)
            print(f"\n📋 Top 5 Results (by composite score):")
            for i, result in enumerate(results[:5]):
                print(f"  {i+1}. inflation={result['inflation']}, k={result['k']}, "
                      f"clusters={result['n_clusters']}, score={result['composite_score']:.3f}, "
                      f"singleton_ratio={result['singleton_ratio']:.3f}")
    
    return best_params, results

def test_optimized_parameters():
    """최적화된 파라미터로 테스트"""
    print(f"\n🧪 Testing Optimized Parameters")
    print("=" * 40)
    
    best_params, _ = optimize_mcl_for_sentences()
    
    if not best_params:
        print("❌ No optimized parameters to test")
        return
    
    # 더 큰 테스트 데이터로 검증
    np.random.seed(123)  # 다른 시드로 검증
    
    # 5개 그룹으로 더 어려운 테스트
    groups = []
    for i in range(5):
        center = np.random.randn(128) * 2
        group_data = np.random.randn(20, 128) * 0.4 + center
        groups.append(group_data)
    
    embeddings = np.vstack(groups)
    true_labels = np.array([i for i in range(5) for _ in range(20)])
    
    print(f"Validation data: {embeddings.shape[0]} embeddings, 5 groups")
    
    # 최적화된 파라미터로 테스트
    mcl = MCLPipeline(
        inflation=best_params['inflation'],
        max_iters=best_params['max_iter']
    )
    
    start_time = time.time()
    mcl.fit(embeddings, k=best_params['k'])
    fit_time = time.time() - start_time
    
    summary = mcl.get_cluster_summary()
    scores = mcl_scoring_function(true_labels, mcl.cluster_labels)
    
    print(f"Validation Results:")
    print(f"  Clusters found: {summary['n_clusters']}")
    print(f"  NMI: {scores['metrics']['nmi']:.3f}")
    print(f"  ARI: {scores['metrics']['ari']:.3f}")
    print(f"  Singleton ratio: {scores['cluster_stats']['pred_singleton_ratio']:.3f}")
    print(f"  Fit time: {fit_time:.3f}s")
    
    # KMeans와 비교
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings)
    kmeans_scores = mcl_scoring_function(true_labels, kmeans_labels)
    
    print(f"\nKMeans comparison:")
    print(f"  NMI: {kmeans_scores['metrics']['nmi']:.3f}")
    print(f"  ARI: {kmeans_scores['metrics']['ari']:.3f}")
    
    if scores['metrics']['nmi'] > 0.5 and scores['metrics']['ari'] > 0.3:
        print("✅ Optimized MCL performs reasonably well!")
    else:
        print("❌ MCL still underperforming compared to KMeans")

if __name__ == "__main__":
    optimize_mcl_for_sentences()
    test_optimized_parameters()
    
    print("\n✅ MCL optimization completed!")