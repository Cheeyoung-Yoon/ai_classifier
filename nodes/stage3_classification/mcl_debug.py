"""
MCL 알고리즘 문제 진단 및 최적화
"""
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

def debug_mcl_algorithm():
    """MCL 알고리즘의 문제점 진단"""
    print("🔍 MCL Algorithm Debug")
    print("=" * 40)
    
    # 잘 분리된 클러스터 데이터 생성
    np.random.seed(42)
    
    # 3개의 명확히 분리된 클러스터
    cluster1 = np.random.randn(20, 50) + np.array([5, 0] + [0]*48)
    cluster2 = np.random.randn(20, 50) + np.array([0, 5] + [0]*48)  
    cluster3 = np.random.randn(20, 50) + np.array([-5, -5] + [0]*48)
    
    embeddings = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.array([0]*20 + [1]*20 + [2]*20)
    
    print(f"Test data: {embeddings.shape[0]} samples, 3 true clusters")
    
    # Step 1: 유사도 매트릭스 확인
    print("\n--- Step 1: Similarity Matrix Analysis ---")
    similarity_matrix = cosine_similarity(embeddings)
    
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Similarity range: {similarity_matrix.min():.3f} to {similarity_matrix.max():.3f}")
    print(f"Average similarity: {similarity_matrix.mean():.3f}")
    
    # 클러스터 내/간 유사도 확인
    within_cluster_sim = []
    between_cluster_sim = []
    
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = similarity_matrix[i, j]
            if true_labels[i] == true_labels[j]:
                within_cluster_sim.append(sim)
            else:
                between_cluster_sim.append(sim)
    
    print(f"Within-cluster similarity: {np.mean(within_cluster_sim):.3f} ± {np.std(within_cluster_sim):.3f}")
    print(f"Between-cluster similarity: {np.mean(between_cluster_sim):.3f} ± {np.std(between_cluster_sim):.3f}")
    
    # Step 2: k-NN 그래프 확인
    print("\n--- Step 2: k-NN Graph Analysis ---")
    for k in [5, 10, 15, 20]:
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # 각 포인트의 이웃들이 같은 클러스터에 있는지 확인
        same_cluster_neighbors = 0
        total_neighbors = 0
        
        for i in range(len(embeddings)):
            neighbors = indices[i][1:]  # 자기 자신 제외
            for neighbor in neighbors:
                total_neighbors += 1
                if true_labels[i] == true_labels[neighbor]:
                    same_cluster_neighbors += 1
        
        accuracy = same_cluster_neighbors / total_neighbors
        print(f"k={k:2d}: Neighbor accuracy = {accuracy:.3f}")
        
        if accuracy > 0.8:
            print(f"  ✅ Good k value: {k}")
        else:
            print(f"  ❌ Poor k value: {k}")

def test_mcl_parameters():
    """MCL 파라미터 테스트"""
    print("\n🔍 MCL Parameter Testing")
    print("=" * 40)
    
    # 테스트 데이터
    np.random.seed(42)
    cluster1 = np.random.randn(15, 30) + np.array([3, 0] + [0]*28)
    cluster2 = np.random.randn(15, 30) + np.array([0, 3] + [0]*28)
    cluster3 = np.random.randn(15, 30) + np.array([-3, -3] + [0]*28)
    
    embeddings = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.array([0]*15 + [1]*15 + [2]*15)
    
    from mcl_pipeline import MCLPipeline
    from evaluation import mcl_scoring_function
    
    # 다양한 파라미터 조합 테스트
    parameter_combinations = [
        {"inflation": 1.5, "k": 8, "max_iters": 50},
        {"inflation": 2.0, "k": 8, "max_iters": 50},
        {"inflation": 3.0, "k": 8, "max_iters": 50},
        {"inflation": 2.0, "k": 5, "max_iters": 50},
        {"inflation": 2.0, "k": 12, "max_iters": 50},
        {"inflation": 2.0, "k": 8, "max_iters": 20},
        {"inflation": 2.0, "k": 8, "max_iters": 100},
    ]
    
    best_score = -1
    best_params = None
    
    for params in parameter_combinations:
        try:
            start_time = time.time()
            mcl = MCLPipeline(inflation=params["inflation"], max_iters=params["max_iters"])
            mcl.fit(embeddings, k=params["k"])
            fit_time = time.time() - start_time
            
            summary = mcl.get_cluster_summary()
            n_clusters = summary["n_clusters"]
            
            # 평가
            scores = mcl_scoring_function(true_labels, mcl.cluster_labels)
            nmi = scores['metrics']['nmi']
            ari = scores['metrics']['ari']
            composite_score = (nmi + ari) / 2
            
            print(f"Params: {params}")
            print(f"  Time: {fit_time:.3f}s, Clusters: {n_clusters}, NMI: {nmi:.3f}, ARI: {ari:.3f}, Score: {composite_score:.3f}")
            
            if composite_score > best_score:
                best_score = composite_score
                best_params = params
                
        except Exception as e:
            print(f"Params: {params} - FAILED: {e}")
    
    print(f"\n🏆 Best parameters: {best_params}")
    print(f"🏆 Best score: {best_score:.3f}")

def test_alternative_clustering():
    """대안 클러스터링 알고리즘과 비교"""
    print("\n🔍 Alternative Clustering Comparison")
    print("=" * 40)
    
    # 테스트 데이터
    np.random.seed(42)
    cluster1 = np.random.randn(20, 50) + np.array([3, 0] + [0]*48)
    cluster2 = np.random.randn(20, 50) + np.array([0, 3] + [0]*48)
    cluster3 = np.random.randn(20, 50) + np.array([-3, -3] + [0]*48)
    
    embeddings = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.array([0]*20 + [1]*20 + [2]*20)
    
    from evaluation import mcl_scoring_function
    
    # 1. KMeans
    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans_labels = kmeans.fit_predict(embeddings)
        kmeans_scores = mcl_scoring_function(true_labels, kmeans_labels)
        print(f"KMeans: NMI={kmeans_scores['metrics']['nmi']:.3f}, ARI={kmeans_scores['metrics']['ari']:.3f}")
    except Exception as e:
        print(f"KMeans failed: {e}")
    
    # 2. DBSCAN
    try:
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(embeddings)
        if len(np.unique(dbscan_labels)) > 1:
            dbscan_scores = mcl_scoring_function(true_labels, dbscan_labels)
            print(f"DBSCAN: NMI={dbscan_scores['metrics']['nmi']:.3f}, ARI={dbscan_scores['metrics']['ari']:.3f}")
        else:
            print("DBSCAN: Failed to find multiple clusters")
    except Exception as e:
        print(f"DBSCAN failed: {e}")
    
    # 3. Spectral Clustering
    try:
        from sklearn.cluster import SpectralClustering
        spectral = SpectralClustering(n_clusters=3, random_state=42)
        spectral_labels = spectral.fit_predict(embeddings)
        spectral_scores = mcl_scoring_function(true_labels, spectral_labels)
        print(f"Spectral: NMI={spectral_scores['metrics']['nmi']:.3f}, ARI={spectral_scores['metrics']['ari']:.3f}")
    except Exception as e:
        print(f"Spectral failed: {e}")

if __name__ == "__main__":
    debug_mcl_algorithm()
    test_mcl_parameters()
    test_alternative_clustering()
    
    print("\n✅ MCL debugging completed!")