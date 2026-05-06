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
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8fafc;
    }
    
    /* Premium Cards */
    .stMetric, .stDataFrame, .stPlotlyChart {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        border: 1px solid #e2e8f0;
        margin-bottom: 1.5rem;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        color: white;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1;
    }
    
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        background-color: #4f46e5;
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #4338ca;
        transform: translateY(-1px);
    }
    
    /* Section Headers */
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
    }
    
    /* Custom Info Box */
    .stAlert {
        border-radius: 12px;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_pipeline():
    from src.pipeline import ResumeMatchingPipeline
    return ResumeMatchingPipeline()

def render_score_gauge(score: float, title: str = "Match Confidence"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 20, "color": "#475569"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar": {"color": "#4f46e5"},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#e2e8f0",
            "steps": [
                {"range": [0, 40], "color": "#fee2e2"},
                {"range": [40, 70], "color": "#fef3c7"},
                {"range": [70, 100], "color": "#dcfce7"},
            ],
        },
    ))
    fig.update_layout(
        height=300, 
        margin=dict(t=80, b=40, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

def render_section_bar(section_scores: dict):
    sections = [s.capitalize() for s in section_scores.keys()]
    scores = [val * 100 for val in section_scores.values()]
    
    fig = px.bar(
        x=scores, y=sections, orientation='h',
        color=scores,
        color_continuous_scale=['#ef4444', '#f59e0b', '#10b981'],
        labels={'x': 'Compatibility %', 'y': 'Section'}
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        height=350,
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(range=[0, 100], gridcolor='#f1f5f9'),
        yaxis=dict(gridcolor='rgba(0,0,0,0)')
    )
    return fig

def main():
    # --- SIDEBAR INPUTS ---
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
        st.title("Kiro Intelligence")
        st.markdown("Upload a candidate profile and job requirements to begin analysis.")
        st.divider()
        
        resume_file = st.file_uploader("📄 Upload Candidate Resume", type=["pdf", "docx", "txt"])
        jd_text = st.text_area("📝 Job Description", height=250, placeholder="Paste JD here...")
        jd_skills_raw = st.text_input("🛠 Key Skills (Optional)", placeholder="e.g. Python, SQL...")
        
        analyze_btn = st.button("🚀 Analyze Alignment")
        
        st.divider()
        st.markdown("### 🧬 System Blueprint")
        with st.expander("Active Novelties (N1-N6)"):
            st.caption("✅ **N1**: Asymmetric Cross-Attention")
            st.caption("✅ **N2**: Dynamic Skill Graph (ESCO)")
            st.caption("✅ **N3**: LLM Weak Labelling")
            st.caption("✅ **N4**: Platt Calibration")
            st.caption("✅ **N5**: Counterfactual Rewriting")
            st.caption("✅ **N6**: Automated Bias Audit")

    # --- MAIN DASHBOARD ---
    if not analyze_btn:
        st.markdown("""
            <div style='text-align: center; padding: 100px;'>
                <img src='https://img.icons8.com/fluency/144/search-in-list.png' width='120'>
                <h1 style='font-size: 3rem;'>Ready for Analysis</h1>
                <p style='font-size: 1.2rem; color: #64748b;'>Upload a resume in the sidebar to generate a deep-intelligence match report.</p>
            </div>
        """, unsafe_allow_html=True)
    
    else:
        if not resume_file or not jd_text:
            st.error("Please provide both a resume and a job description.")
            return

        pipeline = load_pipeline()
        
        with st.spinner("🧠 Computing semantic alignment..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(resume_file.name).suffix) as tmp:
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

        # --- HEADER & NAVIGATION ---
        tabs = st.tabs(["📊 Matching Analysis", "⚖️ Bias & Fairness", "🧬 Technical Deep-Dive"])

        # --- TAB 1: MATCHING ---
        with tabs[0]:
            st.markdown(f"## 📊 Match Insights Report")
            
            mcol1, mcol2 = st.columns([1, 1])
            with mcol1:
                st.plotly_chart(render_score_gauge(result.match_score.overall_score), width='stretch')
            with mcol2:
                st.markdown("### 🎯 Analysis Verdict")
                st.success(result.match_score.calibrated_message)
                st.plotly_chart(render_section_bar(result.match_score.section_scores), width='stretch')

            # --- SKILL MATRIX ---
            st.markdown("### 🛠 Comprehensive Skill Matrix")
            scol1, scol2, scol3 = st.columns(3)
            with scol1:
                st.markdown("<h4 style='color: #10b981;'>✅ Matched</h4>", unsafe_allow_html=True)
                for res_skill, jd_skill, score in result.skill_gaps.matched_skills[:10]:
                    st.write(f"• **{jd_skill}**")
            with scol2:
                st.markdown("<h4 style='color: #ef4444;'>❌ Missing</h4>", unsafe_allow_html=True)
                for skill in result.skill_gaps.missing_skills[:10]:
                    st.write(f"• {skill}")
            with scol3:
                st.markdown("<h4 style='color: #f59e0b;'>⚠️ Weak Match</h4>", unsafe_allow_html=True)
                for res_skill, jd_skill, score in result.skill_gaps.weak_matches[:10]:
                    st.write(f"• {jd_skill} (via *{res_skill}*)")

            # --- COUNTERFACTUAL INSIGHTS ---
            if result.counterfactual:
                st.markdown("---")
                st.markdown("### 💡 Strategic Resume Improvement (N5)")
                cf = result.counterfactual
                impact_color = "#10b981" if cf.delta > 0 else "#ef4444"
                st.markdown(f"""
                    <div style='background-color: white; padding: 1.5rem; border-radius: 12px; border: 1px solid #e2e8f0;'>
                        <p style='color: #64748b; margin-bottom: 0;'>Predicted Score Impact</p>
                        <h2 style='color: {impact_color}; margin-top: 0;'>{cf.original_score:.0f}% → {cf.new_score:.0f}% (+{cf.delta:.1f}%)</h2>
                    </div>
                """, unsafe_allow_html=True)
                icol1, icol2 = st.columns(2)
                with icol1:
                    st.markdown("**Current Phrasing**")
                    st.warning(cf.original_bullet)
                with icol2:
                    st.markdown("**Recommended Optimization**")
                    st.success(cf.rewritten_bullet)

        # --- TAB 2: BIAS AUDIT ---
        with tabs[1]:
            st.markdown("## ⚖️ AI Fairness & Bias Audit (N6)")
            st.markdown("""
                This module runs a real-time audit using institutional and gender-based proxies 
                to ensure the scoring engine is not biased by non-technical factors.
            """)
            
            if st.button("🔍 Run Live Fairness Audit"):
                from src.bias_audit.auditor import BiasAuditor
                
                # Simple wrapper for the auditor
                def score_wrapper(text, jd):
                    # Mock parsing for the auditor
                    sections = {"experience": text, "summary": "", "skills": "", "education": "", "projects": ""}
                    res = pipeline.scorer.compute_score(sections, jd)
                    return res.overall_score
                
                auditor = BiasAuditor(score_wrapper)
                with st.spinner("Injecting identity proxies..."):
                    report = auditor.run_full_audit(" ".join(result.resume_sections.values()), jd_text)
                
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    st.metric("Gender Disparity", f"{report.gender_disparity:.1f}%", delta="Normal" if report.gender_disparity < 10 else "High", delta_color="inverse")
                with bcol2:
                    st.metric("Institution Disparity", f"{report.institution_disparity:.1f}%", delta="Fair" if report.institution_disparity < 10 else "Check", delta_color="inverse")
                
                if report.passed:
                    st.success("✅ AUDIT PASSED: The model shows no statistically significant bias for this resume/JD pair.")
                else:
                    st.warning("⚠️ AUDIT WARNING: Minor scoring variance detected. Review institutional weights.")

        # --- TAB 3: TECHNICAL DEEP-DIVE ---
        with tabs[2]:
            st.markdown("## 🧬 System Architecture Intelligence")
            
            tcol1, tcol2 = st.columns(2)
            
            with tcol1:
                st.markdown("### 🕸 Dynamic Skill Graph (N2)")
                st.markdown("Proving semantic understanding beyond keywords:")
                if jd_skills:
                    sample_skill = jd_skills[0]
                    related = pipeline.skill_graph.get_related_skills(sample_skill, top_k=5)
                    st.write(f"**Seed Skill**: {sample_skill}")
                    st.write("**Learned Neighbors** (ESCO + Co-occurrence):")
                    for s, sim in related:
                        st.progress(sim, text=f"{s} ({sim:.0%})")
            
            with tcol2:
                st.markdown("### 🎯 Section Weighting (N1)")
                st.markdown("Learned importance per resume section:")
                for sec, weight in sorted(result.match_score.section_weights.items(), key=lambda x: -x[1]):
                    st.progress(weight, text=f"{sec.capitalize()}: {weight:.1%}")

if __name__ == "__main__":
    main()
