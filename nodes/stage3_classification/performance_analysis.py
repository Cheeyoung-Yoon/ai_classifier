"""
Performance analysis tool for Stage 3 MCL clustering.
메모리 사용량과 실행 시간을 모니터링합니다.
"""
import time
import psutil
import os
import numpy as np
import pandas as pd

def monitor_memory_usage():
    """현재 메모리 사용량 확인"""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb

def test_mcl_performance():
    """MCL 성능 테스트"""
    print("🔍 MCL Performance Analysis")
    print("=" * 50)
    
    # 메모리 사용량 모니터링
    initial_memory = monitor_memory_usage()
    print(f"Initial memory: {initial_memory:.1f} MB")
    
    # 다양한 크기의 데이터로 테스트
    sizes = [50, 100, 200, 500]
    
    for size in sizes:
        print(f"\n--- Testing with {size} samples ---")
        
        # 데이터 생성
        start_time = time.time()
        embeddings = np.random.randn(size, 128)
        data_gen_time = time.time() - start_time
        
        memory_after_data = monitor_memory_usage()
        print(f"Data generation: {data_gen_time:.3f}s, Memory: {memory_after_data:.1f} MB")
        
        # MCL 테스트
        try:
            from mcl_pipeline import MCLPipeline
            
            start_time = time.time()
            mcl = MCLPipeline(inflation=2.0, max_iters=50)
            mcl_time = time.time() - start_time
            
            memory_after_mcl = monitor_memory_usage()
            print(f"MCL init: {mcl_time:.3f}s, Memory: {memory_after_mcl:.1f} MB")
            
            # Fit 테스트
            start_time = time.time()
            mcl.fit(embeddings, k=min(10, size-1))
            fit_time = time.time() - start_time
            
            memory_after_fit = monitor_memory_usage()
            print(f"MCL fit: {fit_time:.3f}s, Memory: {memory_after_fit:.1f} MB")
            
            # 결과 확인
            summary = mcl.get_cluster_summary()
            print(f"Clusters found: {summary.get('n_clusters', 0)}")
            
            # 메모리 정리
            del mcl, embeddings
            
        except Exception as e:
            print(f"❌ MCL test failed: {e}")
            
        current_memory = monitor_memory_usage()
        memory_diff = current_memory - initial_memory
        print(f"Memory usage: {current_memory:.1f} MB (+{memory_diff:.1f} MB)")
        
        if memory_diff > 100:  # 100MB 이상 증가
            print("⚠️  Potential memory leak detected!")

def test_data_loading_performance():
    """데이터 로딩 성능 테스트"""
    print("\n🔍 Data Loading Performance Analysis")
    print("=" * 50)
    
    # 실제 데이터 파일 테스트
    test_file = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/temp_data/stage2_results/stage2_문4_img_20250918_112537.csv"
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return
    
    initial_memory = monitor_memory_usage()
    print(f"Initial memory: {initial_memory:.1f} MB")
    
    # CSV 로딩 테스트
    start_time = time.time()
    df = pd.read_csv(test_file)
    load_time = time.time() - start_time
    
    memory_after_load = monitor_memory_usage()
    print(f"CSV loading: {load_time:.3f}s, Memory: {memory_after_load:.1f} MB")
    print(f"Data shape: {df.shape}")
    
    # 임베딩 파싱 테스트
    embed_col = 'embed_text_1'
    embeddings = []
    
    start_time = time.time()
    for idx, row in df.iterrows():
        embed_str = row[embed_col]
        if pd.notna(embed_str) and embed_str != '':
            try:
                if isinstance(embed_str, str) and embed_str.startswith('['):
                    embed_array = eval(embed_str)  # 실제로는 ast.literal_eval 사용
                    embeddings.append(embed_array)
                    if len(embeddings) >= 100:  # 제한
                        break
            except:
                continue
                
    parse_time = time.time() - start_time
    memory_after_parse = monitor_memory_usage()
    
    print(f"Embedding parsing: {parse_time:.3f}s, Memory: {memory_after_parse:.1f} MB")
    print(f"Parsed embeddings: {len(embeddings)}")
    
    if len(embeddings) > 0:
        embeddings_array = np.array(embeddings)
        print(f"Embeddings shape: {embeddings_array.shape}")
        
        memory_after_array = monitor_memory_usage()
        print(f"Array conversion: Memory: {memory_after_array:.1f} MB")

def identify_bottlenecks():
    """병목 구간 식별"""
    print("\n🔍 Bottleneck Analysis")
    print("=" * 50)
    
    # 각 단계별 시간 측정
    stages = {
        "Data loading": test_simple_csv_load,
        "Embedding parsing": test_embedding_parsing,
        "MCL clustering": test_mcl_clustering,
        "Evaluation": test_evaluation_performance
    }
    
    for stage_name, test_func in stages.items():
        start_time = time.time()
        start_memory = monitor_memory_usage()
        
        try:
            test_func()
            elapsed = time.time() - start_time
            end_memory = monitor_memory_usage()
            memory_used = end_memory - start_memory
            
            print(f"{stage_name:20}: {elapsed:.3f}s, Memory: +{memory_used:.1f} MB")
            
            if elapsed > 5.0:  # 5초 이상
                print(f"⚠️  {stage_name} is slow!")
            if memory_used > 50:  # 50MB 이상
                print(f"⚠️  {stage_name} uses too much memory!")
                
        except Exception as e:
            print(f"{stage_name:20}: FAILED - {e}")

def test_simple_csv_load():
    """간단한 CSV 로딩 테스트"""
    test_file = "/home/cyyoon/test_area/ai_text_classification/2.langgraph/data/test/temp_data/stage2_results/stage2_문4_img_20250918_112537.csv"
    if os.path.exists(test_file):
        df = pd.read_csv(test_file)
        return len(df)
    return 0

def test_embedding_parsing():
    """임베딩 파싱 테스트"""
    # 가짜 임베딩 문자열
    embed_str = str(np.random.randn(768).tolist())
    embeddings = []
    for _ in range(100):
        embed_array = eval(embed_str)
        embeddings.append(embed_array)
    return len(embeddings)

def test_mcl_clustering():
    """MCL 클러스터링 테스트"""
    from mcl_pipeline import MCLPipeline
    embeddings = np.random.randn(50, 128)
    mcl = MCLPipeline(inflation=2.0, max_iters=20)
    mcl.fit(embeddings, k=10)
    return mcl.get_cluster_summary()['n_clusters']

def test_evaluation_performance():
    """평가 성능 테스트"""
    from evaluation import mcl_scoring_function
    true_labels = np.random.randint(0, 5, size=100)
    pred_labels = np.random.randint(0, 5, size=100)
    scores = mcl_scoring_function(true_labels, pred_labels)
    return scores['metrics']['nmi']

if __name__ == "__main__":
    print("🚀 Performance Analysis Starting...")
    
    try:
        test_mcl_performance()
        test_data_loading_performance()
        identify_bottlenecks()
        
        print("\n✅ Performance analysis completed!")
        
    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
        import traceback
        traceback.print_exc()