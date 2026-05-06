"""
Streamlit UI for AI Resume Matcher.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import os
import sys

# Ensure src is in path
sys.path.insert(0, os.getcwd())

st.set_page_config(
    page_title="Kiro Matcher | Research-Grade AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .card {
        background-color: #1a1c24;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #30363d;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    from src.pipeline import ResumeMatchingPipeline
    return ResumeMatchingPipeline()

def render_score_gauge(score: float, title: str = "Match Probability"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 24, "color": "white"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
            "bar": {"color": "#4facfe"},
            "bgcolor": "#1a1c24",
            "borderwidth": 2,
            "bordercolor": "#30363d",
            "steps": [
                {"range": [0, 40],  "color": "#3d1b1b"},
                {"range": [40, 70], "color": "#3d321b"},
                {"range": [70, 100],"color": "#1b3d24"},
            ],
            "threshold": {
                "line": {"color": "#00f2fe", "width": 4},
                "thickness": 0.75,
                "value": 75,
            },
        },
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "white", 'family': "Inter"},
        height=300,
        margin=dict(t=50, b=20, l=30, r=30)
    )
    return fig

def render_section_bar(section_scores: dict):
    sections = list(section_scores.keys())
    scores = [s * 100 for s in section_scores.values()]
    
    fig = px.bar(
        x=scores, y=sections, orientation='h',
        labels={'x': 'Match Strength %', 'y': 'Section'},
        template="plotly_dark",
        color=scores,
        color_continuous_scale="Viridis"
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(t=20, b=20)
    )
    return fig

def main():
    st.markdown('<div class="main-header">Kiro AI Resume Matcher</div>', unsafe_allow_html=True)
    st.markdown("#### Research-Grade Asymmetric Cross-Attention Engine")
    
    pipeline = load_pipeline()
    
    with st.sidebar:
        st.header("Upload Center")
        resume_file = st.file_uploader("Candidate Resume (PDF/DOCX)", type=["pdf", "docx", "txt"])
        st.markdown("---")
        st.info("System Novelty: N1-N6 Integrated")

    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.markdown("### 📝 Job Description")
        jd_text = st.text_area("Paste the JD requirement here...", height=300)
        jd_skills_raw = st.text_input("Target Skills (comma separated)", placeholder="Python, Pytorch, SQL...")
    
    if st.button("🚀 Run Deep Analysis", type="primary") and resume_file and jd_text:
        with st.spinner("Executing Hierarchical Matching & Counterfactual Rewriting..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(resume_file.name)[1]) as tmp:
                tmp.write(resume_file.read())
                tmp_path = tmp.name
            
            jd_skills = [s.strip() for s in jd_skills_raw.split(",") if s.strip()]
            
            result = pipeline.analyze_from_file(
                resume_path=tmp_path,
                jd_text=jd_text,
                jd_skills=jd_skills or None,
                include_counterfactual=True,
            )
            os.unlink(tmp_path)
            
            # --- Results Layout ---
            st.markdown("---")
            rcol1, rcol2 = st.columns([1, 1])
            
            with rcol1:
                st.plotly_chart(render_score_gauge(result.match_score.overall_score), use_container_width=True)
            
            with rcol2:
                st.markdown("### 🔍 Analysis Verdict")
                st.success(result.match_score.calibrated_message)
                st.plotly_chart(render_section_bar(result.match_score.section_scores), use_container_width=True)

            # --- Skills Grid ---
            st.markdown("### 🛠 Skill Matrix")
            sk1, sk2, sk3 = st.columns(3)
            with sk1:
                st.markdown("##### ✅ Strong Match")
                for r_s, j_s, s in result.skill_gaps.matched_skills[:5]:
                    st.write(f"**{j_s}** (100%)")
            with sk2:
                st.markdown("##### ⚠️ Semantic Near-Miss")
                for r_s, j_s, s in result.skill_gaps.weak_matches[:5]:
                    st.write(f"**{j_s}** matched via *{r_s}* ({s:.0%})")
            with sk3:
                st.markdown("##### ❌ Critical Gaps")
                for s in result.skill_gaps.missing_skills[:5]:
                    st.write(f"**{s}**")

            # --- Counterfactual ---
            if result.counterfactual:
                cf = result.counterfactual
                st.markdown("---")
                st.markdown("### 💡 Actionable Improvement (Delta-Score)")
                st.info(f"Applying this change increases your match score by **{cf.delta:+.1f}%**")
                
                ccol1, ccol2 = st.columns(2)
                with ccol1:
                    st.markdown("**Original Bullet**")
                    st.error(cf.original_bullet)
                with ccol2:
                    st.markdown("**AI-Optimized Version**")
                    st.success(cf.rewritten_bullet)
                
                st.caption(f"Targeting: {cf.jd_requirement}")

if __name__ == "__main__":
    main()
