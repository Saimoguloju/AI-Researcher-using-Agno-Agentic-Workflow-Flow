import streamlit as st
import asyncio
import json
import time
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
import sys
from pathlib import Path

# Add the main code directory to the path
sys.path.append(str(Path(__file__).parent))

# Import your research workflow (assuming your main code is in code.py)
try:
    from code import ResearchWorkflow, ResearchContext, ResearchPhase
except ImportError:
    st.error("Could not import the research workflow. Make sure code.py is in the same directory.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Agno Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .status-running {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    
    .status-complete {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    
    .status-error {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    
    .insight-item {
        background: #f8f9fa;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 3px solid #28a745;
    }
    
    .source-item {
        background: #e9ecef;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 6px;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'research_history' not in st.session_state:
    st.session_state.research_history = []
if 'current_research' not in st.session_state:
    st.session_state.current_research = None
if 'workflow_status' not in st.session_state:
    st.session_state.workflow_status = "idle"

# Helper functions
def run_async_research(workflow, topic, config):
    """Run the async research workflow in a sync context"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(workflow.execute(topic, **config))
        loop.close()
        return result
    except Exception as e:
        st.error(f"Research failed: {str(e)}")
        return None

def format_processing_time(seconds):
    """Format processing time in a human-readable way"""
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        remaining_seconds = seconds % 60
        return f"{int(minutes)}m {remaining_seconds:.0f}s"
    else:
        hours = seconds // 3600
        remaining_minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(remaining_minutes)}m"

def create_quality_metrics_chart(metrics):
    """Create a radar chart for quality metrics"""
    categories = ['Source Quality', 'Academic Ratio', 'Insight Density', 'Verification Rate', 'Confidence']
    values = [
        metrics.get('average_credibility', 0),
        metrics.get('academic_ratio', 0),
        min(metrics.get('insight_density', 0) / 2, 1),  # Normalize to 0-1
        metrics.get('credibility_ratio', 0),
        metrics.get('confidence_adjustment', 0)
    ]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Quality Metrics',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color='rgba(102, 126, 234, 1)')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False,
        title="Research Quality Metrics",
        height=400
    )
    
    return fig

def create_source_distribution_chart(sources):
    """Create a pie chart showing source type distribution"""
    source_types = {}
    for source in sources:
        source_type = source.get('source_type', 'unknown')
        source_types[source_type] = source_types.get(source_type, 0) + 1
    
    fig = px.pie(
        values=list(source_types.values()),
        names=list(source_types.keys()),
        title="Source Type Distribution",
        color_discrete_map={
            'academic': '#28a745',
            'web': '#007bff',
            'unknown': '#6c757d'
        }
    )
    
    return fig

def create_timeline_chart(research_history):
    """Create a timeline chart of research history"""
    if not research_history:
        return None
    
    df = pd.DataFrame([
        {
            'Topic': item['topic'][:50] + '...' if len(item['topic']) > 50 else item['topic'],
            'Date': item['timestamp'],
            'Processing Time': item['processing_time'],
            'Confidence': item['confidence_score'],
            'Sources': item['total_sources']
        }
        for item in research_history
    ])
    
    fig = px.scatter(
        df,
        x='Date',
        y='Processing Time',
        size='Sources',
        color='Confidence',
        hover_data=['Topic'],
        title="Research History Timeline",
        color_continuous_scale='Viridis'
    )
    
    return fig

# Main header
st.markdown("""
<div class="main-header">
    <h1>🔬 Agno Research Assistant</h1>
    <p>Advanced Multi-Agent Research Workflow with AI-Powered Analysis</p>
</div>
""", unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("🛠️ Research Configuration")

# Research topic input
topic = st.sidebar.text_area(
    "Research Topic",
    placeholder="Enter your research topic here...",
    help="Describe what you want to research. Be specific for better results."
)

# Research depth selection
depth = st.sidebar.selectbox(
    "Research Depth",
    ["quick", "standard", "comprehensive"],
    index=2,
    help="Quick: 5 sources, Standard: 8 sources, Comprehensive: 10+ sources"
)

# Number of sources
sources_required = st.sidebar.slider(
    "Number of Sources",
    min_value=5,
    max_value=20,
    value=10,
    help="More sources provide better coverage but take longer to process"
)

# Focus areas
st.sidebar.subheader("🎯 Focus Areas")
focus_areas = []

# Predefined focus area templates
focus_templates = {
    "Technology Impact": [
        "performance metrics",
        "adoption rates",
        "technical challenges",
        "implementation strategies"
    ],
    "Business Analysis": [
        "market trends",
        "cost-benefit analysis",
        "competitive landscape",
        "ROI assessment"
    ],
    "Academic Research": [
        "literature review",
        "methodological approaches",
        "empirical findings",
        "theoretical frameworks"
    ]
}

template_choice = st.sidebar.selectbox(
    "Use Focus Template",
    ["Custom"] + list(focus_templates.keys()),
    help="Select a template or create custom focus areas"
)

if template_choice != "Custom":
    focus_areas = focus_templates[template_choice]
    st.sidebar.info(f"Using {template_choice} template: {', '.join(focus_areas)}")
else:
    # Custom focus areas
    for i in range(4):
        focus_area = st.sidebar.text_input(f"Focus Area {i+1}", key=f"focus_{i}")
        if focus_area:
            focus_areas.append(focus_area)

# Advanced settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    enable_fact_checking = st.checkbox("Enable Fact Checking", value=True)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.1)
    save_report = st.checkbox("Save Report to File", value=True)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📊 Research Dashboard")
    
    # Research status
    status_placeholder = st.empty()
    
    # Progress tracking
    progress_placeholder = st.empty()
    
    # Start research button
    if st.button("🚀 Start Research", type="primary", disabled=not topic):
        if topic:
            st.session_state.workflow_status = "running"
            
            # Initialize workflow
            workflow = ResearchWorkflow()
            
            # Research configuration
            config = {
                "depth": depth,
                "sources_required": sources_required,
                "focus_areas": focus_areas
            }
            
            # Show running status
            with status_placeholder.container():
                st.markdown("""
                <div class="status-box status-running">
                    <h4>🔄 Research In Progress</h4>
                    <p>Your multi-agent research workflow is running...</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Progress bar
            progress_bar = progress_placeholder.progress(0)
            
            # Run research with progress updates
            start_time = time.time()
            
            try:
                # Phase 1: Search
                progress_bar.progress(0.25)
                st.info("Phase 1: Information Gathering...")
                
                # Phase 2: Analysis
                progress_bar.progress(0.5)
                st.info("Phase 2: Analysis and Synthesis...")
                
                # Phase 3: Fact-checking
                progress_bar.progress(0.75)
                st.info("Phase 3: Fact Checking and Verification...")
                
                # Phase 4: Report generation
                progress_bar.progress(0.9)
                st.info("Phase 4: Report Generation...")
                
                # Execute research
                result = run_async_research(workflow, topic, config)
                
                if result:
                    progress_bar.progress(1.0)
                    processing_time = time.time() - start_time
                    
                    # Store results
                    st.session_state.current_research = result
                    st.session_state.workflow_status = "complete"
                    
                    # Add to history
                    research_record = {
                        'topic': topic,
                        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'processing_time': processing_time,
                        'confidence_score': result.confidence_score,
                        'total_sources': len(result.sources),
                        'insights_count': len(result.insights),
                        'quality_metrics': result.quality_metrics
                    }
                    st.session_state.research_history.append(research_record)
                    
                    # Show success status
                    with status_placeholder.container():
                        st.markdown(f"""
                        <div class="status-box status-complete">
                            <h4>✅ Research Completed Successfully!</h4>
                            <p>Processing time: {format_processing_time(processing_time)}</p>
                            <p>Confidence score: {result.confidence_score:.0%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Clear progress bar
                    progress_placeholder.empty()
                    
                    # Save report if enabled
                    if save_report:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"research_report_{timestamp}.md"
                        with open(filename, "w", encoding="utf-8") as f:
                            f.write(result.final_report)
                        st.success(f"Report saved as {filename}")
                else:
                    st.session_state.workflow_status = "error"
                    with status_placeholder.container():
                        st.markdown("""
                        <div class="status-box status-error">
                            <h4>❌ Research Failed</h4>
                            <p>An error occurred during the research process.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    progress_placeholder.empty()
                    
            except Exception as e:
                st.session_state.workflow_status = "error"
                with status_placeholder.container():
                    st.markdown(f"""
                    <div class="status-box status-error">
                        <h4>❌ Research Failed</h4>
                        <p>Error: {str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                progress_placeholder.empty()
    
    # Display current research results
    if st.session_state.current_research and st.session_state.workflow_status == "complete":
        result = st.session_state.current_research
        
        st.header("📈 Research Results")
        
        # Key metrics
        col1_1, col1_2, col1_3, col1_4 = st.columns(4)
        
        with col1_1:
            st.metric("Sources Analyzed", len(result.sources))
        
        with col1_2:
            st.metric("Key Insights", len(result.insights))
        
        with col1_3:
            st.metric("Confidence Score", f"{result.confidence_score:.0%}")
        
        with col1_4:
            academic_count = len([s for s in result.sources if s.get('source_type') == 'academic'])
            st.metric("Academic Sources", f"{academic_count}/{len(result.sources)}")
        
        # Charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            if result.quality_metrics:
                quality_chart = create_quality_metrics_chart(result.quality_metrics)
                st.plotly_chart(quality_chart, use_container_width=True)
        
        with chart_col2:
            source_chart = create_source_distribution_chart(result.sources)
            st.plotly_chart(source_chart, use_container_width=True)
        
        # Tabbed content
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Executive Summary", "💡 Key Insights", "🔍 Fact Checks", "📚 Sources"])
        
        with tab1:
            # Extract executive summary from report
            report_lines = result.final_report.split('\n')
            in_exec_summary = False
            exec_summary = []
            
            for line in report_lines:
                if line.strip() == "## Executive Summary":
                    in_exec_summary = True
                    continue
                elif line.startswith("## ") and in_exec_summary:
                    break
                elif in_exec_summary and line.strip():
                    exec_summary.append(line)
            
            if exec_summary:
                st.markdown('\n'.join(exec_summary))
            else:
                st.write("Executive summary not available.")
        
        with tab2:
            st.subheader("Key Research Insights")
            for i, insight in enumerate(result.insights, 1):
                st.markdown(f"""
                <div class="insight-item">
                    <strong>{i}.</strong> {insight}
                </div>
                """, unsafe_allow_html=True)
        
        with tab3:
            st.subheader("Fact Verification Results")
            if result.fact_checks:
                for check in result.fact_checks:
                    status_icon = "✅" if check.get("status") == "supported" else "⚠️"
                    st.markdown(f"""
                    {status_icon} **{check.get('evidence_strength', 'unknown').title()} Evidence** 
                    (Score: {check.get('verification_score', 0):.2f})
                    
                    {check.get('claim', 'Unknown claim')[:200]}...
                    """)
            else:
                st.info("No fact checks performed.")
        
        with tab4:
            st.subheader("Source Documentation")
            
            # Academic sources
            academic_sources = [s for s in result.sources if s.get('source_type') == 'academic']
            if academic_sources:
                st.write("**Academic Sources:**")
                for i, source in enumerate(academic_sources, 1):
                    authors = ', '.join(source.get('authors', ['Unknown Author']))
                    st.markdown(f"""
                    <div class="source-item">
                        <strong>{i}.</strong> {authors} ({source.get('year', 'n.d.')}). 
                        {source.get('title', 'Unknown Title')}. 
                        <em>{source.get('journal', 'Unknown Journal')}</em>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Web sources
            web_sources = [s for s in result.sources if s.get('source_type') != 'academic']
            if web_sources:
                st.write("**Web Sources:**")
                for i, source in enumerate(web_sources, 1):
                    st.markdown(f"""
                    <div class="source-item">
                        <strong>{i}.</strong> {source.get('title', 'Unknown Title')}
                        <br><small>Credibility: {source.get('credibility_score', 0):.2f}/1.0</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Download report button
        st.download_button(
            label="📄 Download Full Report",
            data=result.final_report,
            file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown"
        )

with col2:
    st.header("📊 Analytics")
    
    # Research history
    if st.session_state.research_history:
        st.subheader("Recent Research")
        
        for item in st.session_state.research_history[-5:]:  # Show last 5
            with st.expander(f"🔍 {item['topic'][:40]}..."):
                st.write(f"**Date:** {item['timestamp']}")
                st.write(f"**Processing Time:** {format_processing_time(item['processing_time'])}")
                st.write(f"**Confidence:** {item['confidence_score']:.0%}")
                st.write(f"**Sources:** {item['total_sources']}")
                st.write(f"**Insights:** {item['insights_count']}")
        
        # Timeline chart
        if len(st.session_state.research_history) > 1:
            timeline_chart = create_timeline_chart(st.session_state.research_history)
            if timeline_chart:
                st.plotly_chart(timeline_chart, use_container_width=True)
    else:
        st.info("No research history yet. Start your first research to see analytics!")
    
    # System information
    st.subheader("🔧 System Status")
    
    try:
        workflow = ResearchWorkflow()
        status = workflow.get_workflow_status()
        
        st.json({
            "Framework": status["framework"],
            "Active Agents": len(status["agents"]),
            "Capabilities": len(status["capabilities"])
        })
        
        with st.expander("Agent Details"):
            for role, agent_name in status["agents"].items():
                st.write(f"**{role.title()}:** {agent_name}")
    
    except Exception as e:
        st.error(f"System status unavailable: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🤖 Powered by Agno Framework | Built with Streamlit</p>
    <p>Advanced Multi-Agent Research System with AI-Powered Quality Assurance</p>
</div>
""", unsafe_allow_html=True)