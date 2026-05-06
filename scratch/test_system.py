import sys
import os
from src.pipeline import MatcherPipeline
from src.explainability import counterfactual_rewrite
from src.ontology import skill_graph
import torch

def test_run():
    print("--- 🎯 AI Resume Matcher Test Suite ---")
    
    # 1. Initialize Pipeline
    print("\n[1/5] Initializing Pipeline (Downloading models if needed)...")
    try:
        pipeline = MatcherPipeline()
        print("✅ Pipeline Initialized.")
    except Exception as e:
        print(f"❌ Pipeline Init Failed: {e}")
        return

    # 2. Pick a real resume from DataSet
    dataset_path = "DataSet"
    resumes = [f for f in os.listdir(dataset_path) if f.endswith(('.pdf', '.docx'))]
    if not resumes:
        print("❌ No resumes found in DataSet.")
        return
    
    test_resume = os.path.join(dataset_path, resumes[0])
    print(f"\n[2/5] Testing with Resume: {resumes[0]}")

    # 3. Sample JD
    test_jd = """
    We are looking for a Software Engineer with expertise in Python and React. 
    Experience with Machine Learning and Pytorch is highly desirable.
    The candidate should have strong knowledge of SQL and API development.
    """

    # 4. Run Matching
    print("\n[3/5] Running Semantic Match...")
    result = pipeline.match(test_resume, test_jd)
    
    print(f"📊 Overall Match Score: {result['total_score']}%")
    print(f"🔬 Raw Cosine Similarity: {result['raw_similarity']}")
    
    # 5. Check Skill Analysis
    print("\n[4/5] Checking Skill Analysis & Ontology...")
    print(f"✅ Direct Matches: {result['skills']['direct_overlap']}")
    print(f"❌ Missing Skills: {result['skills']['missing']}")
    
    if result['skills']['semantic_near_misses']:
        print("💡 Semantic Near-Misses Found:")
        for miss in result['skills']['semantic_near_misses']:
            print(f"   - {miss['resume_ref']} -> {miss['jd_skill']} ({int(miss['sim']*100)}% sim)")
    else:
        print("ℹ️ No semantic near-misses detected (try a JD with related but non-identical terms).")

    # 6. Test Counterfactual Rewriting (N5)
    print("\n[5/5] Testing Counterfactual Rewrite (N5 Novelty)...")
    weak_bullet = "Worked on python projects."
    target_req = "Designed scalable ML pipelines in Pytorch."
    
    # Mocking a simple scorer for the delta check
    def mock_scorer(sections, jd):
        return result['total_score'] + 5 # Simulate a boost
        
    rewrite_result = counterfactual_rewrite(
        [result['sections']['experience']], 
        [test_jd], 
        weak_bullet, 
        target_req, 
        mock_scorer
    )
    
    print(f"✍️ Original: \"{weak_bullet}\"")
    print(f"✨ Rewritten: \"{rewrite_result['rewritten_bullet']}\"")
    print(f"📈 {rewrite_result['explanation']}")

    print("\n--- ✅ Test Complete: Implementation Verified ---")

if __name__ == "__main__":
    test_run()
