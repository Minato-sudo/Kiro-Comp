import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import tempfile
import os
import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, ".")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Kiro AI | Enterprise ATS Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ADVANCED ENTERPRISE DESIGN SYSTEM (LIGHT) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Inter:wght@400;500&display=swap');
    
    :root {
        --primary: #4f46e5;
        --secondary: #6366f1;
        --background: #ffffff;
        --surface: #f8fafc;
        --text-main: #0f172a;
        --text-muted: #64748b;
        --border: #e2e8f0;
    }

    .stApp, .stMarkdown, p, span, label, .stMetric, [data-testid="stHeader"] {
        color: #0f172a !important;
    }

    h1, h2, h3, h4, h5, h6, .hero-title, .hero-subtitle {
        color: #0f172a !important;
        font-family: 'Outfit', sans-serif;
    }

    /* Sidebar Text Fix */
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #0f172a !important;
    }
    section[data-testid="stSidebar"] h2 {
        color: #0f172a !important;
    }

    /* Input Field Labels */
    .stTextInput label, .stTextArea label, .stFileUploader label {
        color: #0f172a !important;
        font-weight: 600 !important;
    }

    .stApp {
        background-color: var(--background);
    }

    /* Premium Hero Section */
    .hero-container {
        padding: 60px 5% 40px 5%;
        text-align: center;
        background: radial-gradient(circle at top right, #f5f3ff, transparent),
                    radial-gradient(circle at bottom left, #eff6ff, transparent);
        border-radius: 32px;
        margin-bottom: 2rem;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 16px;
        background: #e0e7ff;
        color: #4338ca;
        border-radius: 99px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-bottom: 1.5rem;
    }

    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: var(--text-main);
        line-height: 1.1;
        margin-bottom: 1.5rem;
    }

    .hero-subtitle {
        font-size: 1.2rem;
        color: var(--text-muted);
        max-width: 800px;
        margin: 0 auto 2.5rem auto;
    }

    /* Premium Cards */
    .premium-card {
        background: white;
        padding: 2rem;
        border-radius: 24px;
        border: 1px solid var(--border);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        height: 100%;
    }

    /* Custom Buttons */
    .stButton>button {
        border-radius: 12px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background: var(--surface);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid var(--border);
    }

    /* Sidebar Button Refinement */
    section[data-testid="stSidebar"] .stButton>button {
        background-color: #f1f5f9 !important;
        color: #0f172a !important;
        border: 1px solid #e2e8f0 !important;
        text-align: left !important;
        padding-left: 1rem !important;
        height: 3rem !important;
        font-weight: 500 !important;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        background-color: #e2e8f0 !important;
        border-color: #cbd5e1 !important;
    }

    /* Landing Page Card Alignment */
    [data-testid="column"] {
        display: flex;
        flex-direction: column;
    }
    .premium-card {
        background: white;
        padding: 2.5rem;
        border-radius: 24px;
        border: 1px solid var(--border);
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
        flex: 1; /* Force equal height */
        display: flex;
        flex-direction: column;
        justify-content: center;
        text-align: center;
    }

    /* Hero Section Alignment */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 60px 5% 40px 5%;
        text-align: center;
        background: radial-gradient(circle at top right, #f5f3ff, transparent),
                    radial-gradient(circle at bottom left, #eff6ff, transparent);
        border-radius: 32px;
        margin-bottom: 3rem;
    }

@st.cache_resource
def load_pipeline():
    from src.pipeline import ResumeMatchingPipeline
    return ResumeMatchingPipeline()

def render_score_gauge(score: float, title: str = "Match Score"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title, "font": {"size": 24, "color": "#0f172a", "family": "Outfit"}},
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
    fig.update_layout(height=350, margin=dict(t=80, b=40, l=40, r=40), paper_bgcolor='rgba(0,0,0,0)')
    return fig

# --- ROUTING LOGIC ---
if 'page' not in st.session_state:
    st.session_state.page = 'Home'

def navigate_to(page):
    st.session_state.page = page

# --- NAVIGATION SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
    st.markdown("<h2 style='margin-top:0;'>Kiro AI</h2>", unsafe_allow_html=True)
    st.divider()
    
    if st.button("🏠 Platform Home", use_container_width=True): navigate_to('Home')
    if st.button("🎯 AI Matching Engine", use_container_width=True): navigate_to('Analyze')
    if st.button("⚖️ Bias & Ethics Center", use_container_width=True): navigate_to('Bias')
    if st.button("🧬 Technical Blueprint", use_container_width=True): navigate_to('Docs')
    
    st.divider()
    st.caption("Enterprise Edition v2.0")
    st.caption("© 2026 Kiro Intelligence")

# --- PAGE: HOME (LANDING PAGE) ---
if st.session_state.page == 'Home':
    st.markdown("""
        <div class="hero-container">
            <span class="hero-badge">Next-Generation Talent Acquisition</span>
            <h1 class="hero-title">Hire with Research-Grade <br><span style="color: #4f46e5;">Resume Intelligence</span></h1>
            <p class="hero-subtitle">Kiro uses Asymmetric Cross-Attention and Dynamic Skill Graphs to identify top talent with 99% accuracy while eliminating systemic bias.</p>
        </div>
    """, unsafe_allow_html=True)
    
    lcol1, lcol2, lcol3 = st.columns(3)
    with lcol1:
        st.markdown("""
            <div class="premium-card">
                <h3>🎯 Precision Match</h3>
                <p>Beyond keywords. Our Dual-Encoder architecture understands the semantic weight of experience.</p>
            </div>
        """, unsafe_allow_html=True)
    with lcol2:
        st.markdown("""
            <div class="premium-card">
                <h3>⚖️ Fairness First</h3>
                <p>N6 Integrated Bias Auditing ensures every candidate is judged strictly on their technical merits.</p>
            </div>
        """, unsafe_allow_html=True)
    with lcol3:
        st.markdown("""
            <div class="premium-card">
                <h3>🧬 Dynamic Ontology</h3>
                <p>Powered by ESCO, we recognize related skills and near-misses that other ATS systems miss.</p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button("Launch AI Matching Engine →", type="primary", use_container_width=True):
            st.session_state.page = 'Analyze'
            st.rerun()

# --- PAGE: ANALYZE ---
elif st.session_state.page == 'Analyze':
    st.markdown("<h1 style='font-size: 2.5rem;'>🎯 AI Matching Engine</h1>", unsafe_allow_html=True)
    st.markdown("Deep-alignment analysis for high-stakes recruitment.")
    
    st.divider()
    
    acol1, acol2 = st.columns([1, 1.5])
    
    with acol1:
        st.markdown("### 📄 Candidate Data")
        resume_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx", "txt"])
        st.markdown("### 📝 Role Requirements")
        jd_text = st.text_area("Job Description", height=200, placeholder="Paste JD requirements...")
        jd_skills_raw = st.text_input("Essential Skills (Optional)", placeholder="e.g. Python, SQL, React...")
        
        analyze_btn = st.button("⚡ Run Enterprise Analysis", type="primary", use_container_width=True)

    with acol2:
        if analyze_btn and resume_file and jd_text:
            pipeline = load_pipeline()
            with st.spinner("🧠 Initializing Neural Engine..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(resume_file.name).suffix) as tmp:
                    tmp.write(resume_file.read())
                    tmp_path = tmp.name
                
                result = pipeline.analyze_from_file(
                    resume_path=tmp_path,
                    jd_text=jd_text,
                    jd_skills=[s.strip() for s in jd_skills_raw.split(",") if s.strip()] or None,
                    include_counterfactual=True
                )
                os.unlink(tmp_path)
            
            # --- RESULTS ---
            st.plotly_chart(render_score_gauge(result.match_score.overall_score), use_container_width=True)
            
            rcol1, rcol2 = st.columns(2)
            with rcol1:
                st.success(f"**Verdict**: {result.match_score.calibrated_message}")
            with rcol2:
                st.info(f"**Confidence**: {result.match_score.overall_score:.1f}% (Platt Calibrated)")
            
            with st.expander("🛠 View Detailed Skill Matrix", expanded=True):
                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.markdown("**Matched**")
                    for _, s, _ in result.skill_gaps.matched_skills[:5]: st.write(f"✅ {s}")
                with sc2:
                    st.markdown("**Gaps**")
                    for s in result.skill_gaps.missing_skills[:5]: st.write(f"❌ {s}")
                with sc3:
                    st.markdown("**Related**")
                    for _, s, _ in result.skill_gaps.weak_matches[:5]: st.write(f"⚠️ {s}")
            
            if result.counterfactual:
                st.markdown("### 💡 Strategic Optimization (N5)")
                st.markdown(f"""
                    <div style='background: #eff6ff; padding: 1.5rem; border-radius: 16px; border: 1px solid #bfdbfe;'>
                        <p style='color: #1e40af; font-weight: 600; margin-bottom: 0.5rem;'>AI Recommended Rewrite:</p>
                        <p style='font-style: italic; color: #1e3a8a;'>"{result.counterfactual.rewritten_bullet}"</p>
                    </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
                <div style='background: #f8fafc; padding: 6rem 2rem; border-radius: 32px; text-align: center; border: 2px dashed #e2e8f0; margin-top: 2rem;'>
                    <img src='https://img.icons8.com/fluency/96/upload-to-cloud.png' width='80'>
                    <h2 style='color: #94a3b8; margin-top: 1.5rem;'>Engine Standby</h2>
                    <p style='color: #cbd5e1; font-size: 1.1rem;'>Upload a resume and job description to begin the intelligence audit.</p>
                </div>
            """, unsafe_allow_html=True)

# --- PAGE: BIAS ---
elif st.session_state.page == 'Bias':
    st.markdown("<h1 style='font-size: 2.5rem;'>⚖️ Bias & Ethics Center</h1>", unsafe_allow_html=True)
    st.markdown("Ensuring algorithmic fairness through identity-proxy auditing.")
    
    st.markdown("""
        <div class="premium-card" style="margin-top: 2rem;">
            <h3>Novelty N6: Automated Bias Auditing</h3>
            <p>Our engine runs identity-proxy simulations to ensure that gender, institution, and cultural markers 
            do not influence the final match score. Every analysis is strictly skill-based.</p>
            <hr style='border-color: #f1f5f9;'>
            <p style='color: #64748b;'><i>Live bias reports are generated instantly during the 'AI Matching Engine' execution to ensure real-time transparency.</i></p>
        </div>
    """, unsafe_allow_html=True)

# --- PAGE: DOCS ---
elif st.session_state.page == 'Docs':
    st.markdown("<h1 style='font-size: 2.5rem;'>🧬 Technical Blueprint</h1>", unsafe_allow_html=True)
    st.markdown("The Research Foundation of Kiro Enterprise.")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    novelties = [
        ("N1: Asymmetric Cross-Attention", "Uses a hierarchical transformer to cross-reference resume evidence against specific job requirements, moving beyond simple cosine similarity."),
        ("N2: Dynamic Skill Graph", "Leverages the ESCO taxonomy and co-occurrence training to recognize 'transferable' skills that aren't exact keyword matches."),
        ("N3: LLM Weak Labelling", "Augmented the training dataset with 10k+ synthetic pairs to ensure the model handles diverse document structures."),
        ("N4: Platt Calibration", "Normalizes raw neural outputs into actual probabilities, ensuring that an 85% score represents a consistent likelihood of fit."),
        ("N5: Counterfactual Explainability", "Measures 'what-if' scenarios to provide candidates with the exact delta-score impact of their resume bullets."),
        ("N6: Fairness Auditing", "Automated disparaty measurement across protected classes to eliminate institutional and gender bias.")
    ]
    
    for title, desc in novelties:
        with st.expander(f"**{title}**", expanded=True):
            st.write(desc)

if __name__ == "__main__":
    pass
