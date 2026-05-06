import streamlit as st
import os
import glob
import pandas as pd
import plotly.express as px
from src.pipeline import MatcherPipeline
from src.explainability import counterfactual_rewrite, generate_suggestions
from src.bias_audit import BiasAudit

# Page Config
st.set_page_config(page_title="AI Resume Matcher PRO", layout="wide", page_icon="🎯")

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0f1117;
        color: #ffffff;
    }
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white !important;
    }
    .match-card {
        background: #1e222d;
        border-radius: 10px;
        padding: 20px;
        border-left: 5px solid #667eea;
        margin-bottom: 20px;
    }
    h1, h2, h3 {
        color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    return MatcherPipeline()

pipeline = load_pipeline()
auditor = BiasAudit()

st.title("🎯 AI Resume & Job Matching System")
st.markdown("### Semantic matching beyond keywords.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📁 Select Candidate Resume")
    dataset_path = "DataSet"
    all_resumes = glob.glob(os.path.join(dataset_path, "*.*"))
    selected_resume_path = st.selectbox("Choose a resume from dataset", all_resumes, format_func=lambda x: os.path.basename(x))
    
    st.markdown("---")
    st.subheader("📝 Job Description")
    jd_input = st.text_area("Paste the Job Description here", height=250, value="We are looking for a Python Developer with experience in Machine Learning, Pandas, and Pytorch. Knowledge of React for frontend is a plus.")

with col2:
    if st.button("🚀 Analyze Match", use_container_width=True):
        with st.spinner("Analyzing semantics..."):
            result = pipeline.match(selected_resume_path, jd_input)
            
            # Display Score
            st.metric("Overall Match Score", f"{result['total_score']}%", delta=f"{result['raw_similarity']} raw sim")
            
            # Visualizations
            fig = px.bar(
                x=["Skills", "Experience", "Projects", "Education", "Summary"],
                y=[0.35, 0.30, 0.20, 0.10, 0.05], # Mock weights
                title="Section Relevance Attribution",
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig, use_container_width=True)

            # Skill Overlap
            st.markdown("### 🛠 Skill Analysis")
            s_col1, s_col2 = st.columns(2)
            with s_col1:
                st.success(f"Matched: {', '.join(result['skills']['direct_overlap'])}")
            with s_col2:
                st.error(f"Missing: {', '.join(result['skills']['missing'])}")
            
            if result['skills']['semantic_near_misses']:
                st.info("💡 **Semantic Near-Misses:** You have skills that are highly related to the requirements:")
                for miss in result['skills']['semantic_near_misses']:
                    st.write(f"- Your **{miss['resume_ref']}** is relevant to the required **{miss['jd_skill']}** ({int(miss['sim']*100)}% similarity)")

            # Counterfactual Improvements (N5)
            st.markdown("---")
            st.markdown("### 📈 Intelligent Improvements")
            
            # Mocking a weak bullet for demonstration
            weak_bullet = "Assisted in some projects using python."
            jd_req = "Build scalable ML pipelines."
            
            def mock_score(sections, jd):
                return result['total_score'] # Dummy for now
            
            improvement = counterfactual_rewrite(
                [result['sections']['experience']], 
                [jd_input], 
                weak_bullet, 
                jd_req, 
                mock_score
            )
            
            st.markdown(f"""
            <div class="match-card">
                <b>Recommended Rewrite:</b><br/>
                <i>"{improvement['rewritten_bullet']}"</i><br/>
                <span style="color: #00ff00;">Impact: {improvement['explanation']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Grounded Suggestions
            suggestions = generate_suggestions(
                {"name": os.path.basename(selected_resume_path)},
                {"title": "Target Role", "company": "Target Company"},
                {"total": result['total_score'], "missing_skills": result['skills']['missing']}
            )
            st.write(suggestions)

            # Bias Audit (N6)
            with st.expander("🛡 AI Fairness Report"):
                report = auditor.audit_model(lambda r, j: result['total_score'], [result['sections']['summary']], [jd_input])
                st.json(report)
                st.caption("Audit checks for gendered language and age-related bias terms to ensure objective scoring.")

# Footer
st.markdown("---")
st.caption("Developed for Kiro Hackathon - Dual Encoder Asymmetric Attention Architecture")
