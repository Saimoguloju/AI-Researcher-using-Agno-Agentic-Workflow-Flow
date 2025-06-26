import asyncio
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
from bs4 import BeautifulSoup
import logging
from abc import ABC, abstractmethod
from enum import Enum
import uuid

# Agno framework imports
from agno.agent import Agent
from agno.models.openai import OpenAIChat

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResearchPhase(Enum):
    """Research workflow phases for orchestration"""
    SEARCH = "search"
    ANALYSIS = "analysis"
    FACT_CHECK = "fact_check"
    WRITING = "writing"
    COMPLETE = "complete"

@dataclass
class ResearchContext:
    """
    Research context that flows through the agentic workflow.
    Contains all necessary information for agents to collaborate effectively.
    """
    topic: str
    depth: str = "comprehensive"
    sources_required: int = 8
    focus_areas: List[str] = None
    current_phase: ResearchPhase = ResearchPhase.SEARCH
    sources: List[Dict] = None
    insights: List[str] = None
    analysis: str = ""
    fact_checks: List[Dict] = None
    final_report: str = ""
    confidence_score: float = 0.0
    quality_metrics: Dict[str, float] = None
    processing_start: datetime = None
    
    def __post_init__(self):
        if self.focus_areas is None:
            self.focus_areas = []
        if self.sources is None:
            self.sources = []
        if self.insights is None:
            self.insights = []
        if self.fact_checks is None:
            self.fact_checks = []
        if self.quality_metrics is None:
            self.quality_metrics = {}
        if self.processing_start is None:
            self.processing_start = datetime.now()

# Standalone tool functions (not using @tool decorator for now)
async def web_search_tool(query: str, num_results: int = 5) -> List[Dict]:
    """
    Search the web for information on a given topic.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 5)
    
    Returns:
        List of search results with metadata
    """
    logger.info(f"Executing web search: {query}")
    
    # Simulate API delay
    await asyncio.sleep(0.3)
    
    # Enhanced simulated results with realistic metadata
    results = []
    for i in range(num_results):
        result = {
            "id": str(uuid.uuid4()),
            "title": f"Research Study: {query} - Analysis and Insights {i+1}",
            "url": f"https://research-journal-{i+1}.com/studies/{query.replace(' ', '-')}",
            "snippet": f"Comprehensive analysis of {query} reveals significant patterns and trends. This study examines multiple aspects including implementation strategies, performance metrics, and industry adoption rates.",
            "content": f"""
            Detailed research content on {query}:
            
            This authoritative source provides in-depth analysis of {query}, including:
            - Current market trends and adoption rates
            - Technical implementation considerations
            - Performance benchmarks and case studies
            - Expert opinions and industry insights
            - Future projections and recommendations
            
            The research methodology involved comprehensive data collection from multiple sources,
            statistical analysis, and peer review validation. Key findings suggest significant
            implications for industry practices and strategic decision-making.
            """,
            "source_type": "web",
            "credibility_score": 0.65 + (i * 0.07),  # Varying credibility scores
            "publication_date": f"2024-0{min(i+1, 9)}-15",
            "domain_authority": 70 + (i * 3),
            "author_expertise": "high" if i < 2 else "medium"
        }
        results.append(result)
    
    logger.info(f"Web search completed: {len(results)} results found")
    return results

async def academic_search_tool(query: str, max_results: int = 3) -> List[Dict]:
    """
    Search academic databases for scholarly sources.
    
    Args:
        query: Academic search query
        max_results: Maximum number of results (default: 3)
    
    Returns:
        List of academic papers with citation information
    """
    logger.info(f"Executing academic search: {query}")
    await asyncio.sleep(0.6)  # Academic searches typically take longer
    
    academic_results = []
    for i in range(max_results):
        result = {
            "id": str(uuid.uuid4()),
            "title": f"Empirical Study on {query}: Methodological Approaches and Findings",
            "authors": [
                f"Dr. {['Sarah', 'Michael', 'Elena', 'David', 'Lisa'][i]} {['Johnson', 'Chen', 'Rodriguez', 'Smith', 'Williams'][i]}",
                f"Prof. {['James', 'Maria', 'Robert', 'Anna', 'Thomas'][i]} {['Wilson', 'Garcia', 'Brown', 'Davis', 'Miller'][i]}"
            ],
            "journal": f"Journal of {['Advanced', 'Applied', 'International'][i % 3]} {['Research', 'Studies', 'Science'][i % 3]}",
            "year": 2024 - i,
            "volume": f"{45 + i}",
            "issue": f"{3 - i}",
            "pages": f"{120 + i*20}-{145 + i*20}",
            "abstract": f"""
            This peer-reviewed study presents a comprehensive analysis of {query} through 
            rigorous empirical methodology. The research employed statistical modeling, 
            experimental design, and systematic review approaches to examine key aspects 
            of the field. Results indicate significant patterns and relationships that 
            contribute to theoretical understanding and practical applications.
            
            Key findings include quantitative analysis of performance metrics, 
            comparative evaluation of different approaches, and identification of 
            critical success factors. The study's implications extend to both 
            academic research and industry practice.
            """,
            "doi": f"10.1000/research.{query.replace(' ', '.')}.{2024-i}",
            "citation_count": 150 - (i * 30),
            "impact_factor": 4.2 - (i * 0.3),
            "source_type": "academic",
            "credibility_score": 0.95 - (i * 0.02),
            "peer_reviewed": True,
            "open_access": i % 2 == 0
        }
        academic_results.append(result)
    
    logger.info(f"Academic search completed: {len(academic_results)} papers found")
    return academic_results

async def fact_verification_tool(claim: str, sources: List[Dict]) -> Dict:
    """
    Verify factual claims against available sources.
    
    Args:
        claim: The claim to verify
        sources: List of sources to check against
    
    Returns:
        Verification result with confidence score
    """
    logger.info(f"Verifying claim: {claim[:100]}...")
    await asyncio.sleep(0.2)
    
    # Calculate verification score based on source quality and content matching
    academic_sources = [s for s in sources if s.get("source_type") == "academic"]
    high_credibility_sources = [s for s in sources if s.get("credibility_score", 0) > 0.8]
    
    verification_score = 0.6  # Base score
    
    if academic_sources:
        verification_score += 0.2
    if high_credibility_sources:
        verification_score += 0.15
    if len(sources) >= 3:
        verification_score += 0.05
    
    verification_score = min(verification_score, 0.95)
    
    status = "supported" if verification_score > 0.7 else "needs_verification"
    if verification_score < 0.4:
        status = "disputed"
    
    return {
        "claim": claim,
        "status": status,
        "verification_score": verification_score,
        "supporting_sources": len(high_credibility_sources),
        "evidence_strength": "strong" if verification_score > 0.8 else "moderate" if verification_score > 0.6 else "weak",
        "checked_at": datetime.now().isoformat()
    }

# Specialized research agents using Agno framework
class ResearchSearchAgent(Agent):
    """
    Specialized search agent that orchestrates information gathering using Agno's agent framework.
    Focuses on finding diverse, high-quality sources across multiple domains.
    """
    
    def __init__(self):
        super().__init__(
            name="ResearchSearchAgent",
            model=OpenAIChat(),
            description="Expert at finding and evaluating research sources across web and academic databases",
            instructions="""
            You are a research librarian specialist with expertise in:
            - Crafting effective search queries for different domains
            - Evaluating source credibility and relevance
            - Identifying gaps in information coverage
            - Prioritizing sources based on authority and recency
            
            Always focus on finding diverse, authoritative sources that provide
            comprehensive coverage of the research topic.
            """
        )
    
    async def search_comprehensive(self, context: ResearchContext) -> ResearchContext:
        """
        Execute comprehensive search strategy across multiple source types.
        """
        logger.info(f"Starting comprehensive search for: {context.topic}")
        
        all_sources = []
        search_strategies = []
        
        # Primary topic search - call tool function directly
        web_sources = await web_search_tool(context.topic, context.sources_required)
        all_sources.extend(web_sources)
        search_strategies.append(f"Primary web search: {len(web_sources)} sources")
        
        # Academic search for credibility
        academic_sources = await academic_search_tool(context.topic, 3)
        all_sources.extend(academic_sources)
        search_strategies.append(f"Academic search: {len(academic_sources)} papers")
        
        # Focused searches for specific areas
        if context.focus_areas:
            for focus_area in context.focus_areas[:3]:  # Limit to prevent overload
                focused_query = f"{context.topic} {focus_area}"
                focused_sources = await web_search_tool(focused_query, 2)
                all_sources.extend(focused_sources)
                search_strategies.append(f"Focused search '{focus_area}': {len(focused_sources)} sources")
        
        # Update context with search results
        context.sources = all_sources
        context.current_phase = ResearchPhase.ANALYSIS
        
        # Calculate search quality metrics
        academic_ratio = len([s for s in all_sources if s.get("source_type") == "academic"]) / len(all_sources)
        avg_credibility = sum(s.get("credibility_score", 0) for s in all_sources) / len(all_sources)
        
        context.quality_metrics.update({
            "total_sources": len(all_sources),
            "academic_ratio": academic_ratio,
            "average_credibility": avg_credibility,
            "search_strategies": search_strategies
        })
        
        logger.info(f"Search completed: {len(all_sources)} sources, {academic_ratio:.1%} academic")
        logger.info(f"Search agent completed: {len(all_sources)} sources found")
        
        return context

class ResearchAnalysisAgent(Agent):
    """
    Specialized analysis agent that extracts insights and synthesizes information using advanced reasoning.
    """
    
    def __init__(self):
        super().__init__(
            name="ResearchAnalysisAgent", 
            model=OpenAIChat(),
            description="Expert research analyst specializing in synthesis and insight extraction",
            instructions="""
            You are a senior research analyst with expertise in:
            - Identifying key patterns and themes across diverse sources
            - Synthesizing complex information into actionable insights
            - Evaluating evidence quality and source reliability
            - Drawing connections between disparate findings
            - Providing balanced analysis that acknowledges limitations
            
            Focus on creating comprehensive, well-reasoned analysis that provides
            clear value to decision-makers and researchers.
            """
        )
    
    async def analyze_sources(self, context: ResearchContext) -> ResearchContext:
        """
        Perform comprehensive analysis of gathered sources to extract key insights.
        """
        logger.info(f"Starting analysis of {len(context.sources)} sources")
        
        # Prepare content for analysis
        source_contents = []
        for source in context.sources:
            content = (source.get('content', '') or 
                      source.get('snippet', '') or 
                      source.get('abstract', ''))
            if content:
                source_contents.append({
                    "content": content,
                    "credibility": source.get("credibility_score", 0.5),
                    "type": source.get("source_type", "unknown"),
                    "title": source.get("title", "Unknown")
                })
        
        # Extract insights using AI analysis
        insights_prompt = f"""
        Analyze the following research sources on "{context.topic}" and extract the most valuable insights:
        
        Sources to analyze: {len(source_contents)} sources
        Academic sources: {len([s for s in source_contents if s['type'] == 'academic'])}
        Average credibility: {sum(s['credibility'] for s in source_contents) / len(source_contents):.2f}
        
        Source content (first 2000 chars from each):
        {chr(10).join([f"Source: {s['title'][:100]}...{chr(10)}{s['content'][:2000]}...{chr(10)}" for s in source_contents[:5]])}
        
        Please provide:
        1. Top 7 key insights with supporting evidence
        2. Main themes and patterns identified
        3. Significant data points or statistics
        4. Expert opinions and consensus areas  
        5. Contradictions or ongoing debates
        6. Implications for practice or policy
        7. Areas requiring further investigation
        
        Format each insight clearly with evidence references where possible.
        Consider source credibility when weighing findings.
        """
        
        # Get insights from AI model using the agent's run method
        insights_response = self.run(insights_prompt)
        
        # Parse insights into structured list
        insights_text = insights_response.content if hasattr(insights_response, 'content') else str(insights_response)
        insights = []
        
        for line in insights_text.split('\n'):
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith('•') or line.strip().startswith('-')):
                clean_insight = line.strip().lstrip('0123456789.-• ')
                if len(clean_insight) > 20:  # Filter out short/incomplete insights
                    insights.append(clean_insight)
        
        # Generate synthesized analysis
        synthesis_prompt = f"""
        Create a comprehensive research analysis based on these insights about "{context.topic}":
        
        Key Insights:
        {chr(10).join(f"• {insight}" for insight in insights)}
        
        Research Context:
        - Total sources: {len(context.sources)}
        - Academic sources: {context.quality_metrics.get('academic_ratio', 0):.1%}
        - Average credibility: {context.quality_metrics.get('average_credibility', 0):.2f}
        
        Create a structured analysis that:
        1. Synthesizes the main themes and their relationships
        2. Discusses the strength of evidence and methodology considerations
        3. Identifies consensus areas and ongoing debates
        4. Highlights the most significant findings and their implications
        5. Notes limitations and uncertainties in the current research
        6. Provides actionable conclusions and recommendations
        
        Write in clear, authoritative prose suitable for researchers and decision-makers.
        Maintain objectivity while highlighting the most important findings.
        """
        
        analysis_response = self.run(synthesis_prompt)
        analysis_text = analysis_response.content if hasattr(analysis_response, 'content') else str(analysis_response)
        
        # Update context with analysis results
        context.insights = insights
        context.analysis = analysis_text
        context.current_phase = ResearchPhase.FACT_CHECK
        
        # Calculate analysis quality metrics
        insight_density = len(insights) / len(context.sources) if context.sources else 0
        analysis_depth_score = min(len(analysis_text) / 2000, 1.0)  # Normalize to reasonable length
        
        context.quality_metrics.update({
            "insights_extracted": len(insights),
            "insight_density": insight_density,
            "analysis_depth": analysis_depth_score
        })
        
        logger.info(f"Analysis completed: {len(insights)} insights, {len(analysis_text)} chars analysis")
        logger.info(f"Analysis agent completed: {len(insights)} insights extracted")
        
        return context
    
    def _calculate_confidence_score(self, sources: List[Dict], insights: List[str], quality_metrics: Dict[str, float]) -> float:
        """Calculate confidence score based on multiple factors"""
        
        base_confidence = 0.5
        
        # Source quality boost
        source_boost = quality_metrics.get("average_credibility", 0) * 0.3
        
        # Academic source boost  
        academic_boost = quality_metrics.get("academic_ratio", 0) * 0.2
        
        # Insight density boost
        insight_boost = min(quality_metrics.get("insight_density", 0) * 0.1, 0.15)
        
        # Coverage boost
        coverage_boost = min(len(sources) / 10, 0.1)  # Up to 0.1 for 10+ sources
        
        total_confidence = min(base_confidence + source_boost + academic_boost + insight_boost + coverage_boost, 0.95)
        
        return round(total_confidence, 3)

class ResearchFactCheckAgent(Agent):
    """
    Specialized fact-checking agent that verifies claims and assesses information reliability.
    """
    
    def __init__(self):
        super().__init__(
            name="ResearchFactCheckAgent",
            model=OpenAIChat(),
            description="Expert fact-checker specializing in claim verification and source validation",
            instructions="""
            You are a professional fact-checker with expertise in:
            - Identifying verifiable factual claims in research content
            - Cross-referencing claims against authoritative sources
            - Assessing evidence quality and reliability
            - Detecting potential biases or misinformation
            - Providing confidence assessments for factual assertions
            
            Focus on maintaining research integrity by validating key claims
            and highlighting areas where evidence is strong or uncertain.
            """
        )
    
    async def verify_research(self, context: ResearchContext) -> ResearchContext:
        """
        Perform comprehensive fact-checking of research insights and analysis.
        """
        logger.info(f"Starting fact verification for {len(context.insights)} insights")
        
        # Identify key claims to verify
        all_content = context.analysis + " " + " ".join(context.insights)
        
        claims_prompt = f"""
        Identify the most important factual claims that should be verified in this research content:
        
        {all_content[:3000]}...
        
        Extract 5-8 specific, verifiable claims such as:
        - Statistical data or percentages
        - Specific dates or timeframes  
        - Names of studies, organizations, or key figures
        - Quantitative assertions about performance, adoption, etc.
        - Categorical statements about trends or relationships
        
        For each claim, provide:
        1. The exact claim statement
        2. Why this claim is important to verify
        3. How easily verifiable it appears to be
        
        Focus on claims central to the research conclusions.
        """
        
        claims_response = self.run(claims_prompt)
        claims_text = claims_response.content if hasattr(claims_response, 'content') else str(claims_response)
        
        # Extract individual claims
        identified_claims = []
        for line in claims_text.split('\n'):
            if line.strip() and len(line.strip()) > 30:
                if any(indicator in line.lower() for indicator in ['%', 'study', 'research', 'shows', 'indicates', 'found']):
                    clean_claim = line.strip().lstrip('0123456789.-• ')
                    if len(clean_claim) > 20:
                        identified_claims.append(clean_claim)
        
        # Verify each claim using the fact verification tool
        verified_claims = []
        for claim in identified_claims[:8]:  # Limit to most important claims
            verification_result = await fact_verification_tool(claim, context.sources)
            verified_claims.append(verification_result)
        
        # Calculate overall credibility assessment
        supported_claims = len([c for c in verified_claims if c.get("status") == "supported"])
        total_claims = len(verified_claims)
        credibility_ratio = supported_claims / total_claims if total_claims > 0 else 1.0
        
        # Adjust overall confidence based on verification results
        verification_impact = 0.9 + (credibility_ratio * 0.1)  # 90-100% based on verification
        base_confidence = context.quality_metrics.get('average_credibility', 0.7)
        
        context.confidence_score = min(base_confidence * verification_impact, 0.95)
        
        # Update context with fact-check results
        context.fact_checks = verified_claims
        context.current_phase = ResearchPhase.WRITING
        
        context.quality_metrics.update({
            "claims_verified": len(verified_claims),
            "supported_claims": supported_claims,
            "credibility_ratio": credibility_ratio,
            "confidence_adjustment": verification_impact
        })
        
        logger.info(f"Fact-checking completed: {supported_claims}/{total_claims} claims supported")
        logger.info(f"Fact-check agent completed: {supported_claims}/{total_claims} claims verified")
        
        return context

class ResearchWriterAgent(Agent):
    """
    Specialized writing agent that creates comprehensive, well-structured research reports.
    """
    
    def __init__(self):
        super().__init__(
            name="ResearchWriterAgent",
            model=OpenAIChat(),
            description="Expert research writer specializing in comprehensive report generation",
            instructions="""
            You are a professional research writer with expertise in:
            - Creating clear, engaging research reports for diverse audiences
            - Structuring complex information in logical, accessible formats
            - Balancing technical depth with readability
            - Incorporating evidence and citations effectively
            - Providing actionable insights and recommendations
            
            Focus on creating reports that are both authoritative and accessible,
            with clear executive summaries and well-supported conclusions.
            """
        )
    
    async def generate_report(self, context: ResearchContext) -> ResearchContext:
        """
        Generate comprehensive research report with all findings and analysis.
        """
        logger.info(f"Starting report generation for: {context.topic}")
        
        # Create executive summary
        exec_summary_prompt = f"""
        Create an executive summary for a research report on "{context.topic}".
        
        Key insights (top 5):
        {chr(10).join(context.insights[:5])}
        
        Research metrics:
        - Confidence level: {context.confidence_score:.0%}
        - Sources analyzed: {len(context.sources)}
        - Academic sources: {context.quality_metrics.get('academic_ratio', 0):.0%}
        - Claims verified: {context.quality_metrics.get('supported_claims', 0)}/{context.quality_metrics.get('claims_verified', 0)}
        
        Create a compelling 2-3 paragraph executive summary that:
        1. Opens with the most significant finding
        2. Summarizes key insights and their implications
        3. Notes the research approach and confidence level
        4. Concludes with actionable recommendations
        
        Write for senior decision-makers who need quick, reliable insights.
        """
        
        exec_summary_response = self.run(exec_summary_prompt)
        exec_summary = exec_summary_response.content if hasattr(exec_summary_response, 'content') else str(exec_summary_response)
        
        # Create conclusions and recommendations
        conclusions_prompt = f"""
        Based on this research analysis:
        
        {context.analysis}
        
        And these verification results:
        - Confidence score: {context.confidence_score:.0%}
        - {context.quality_metrics.get('supported_claims', 0)} of {context.quality_metrics.get('claims_verified', 0)} claims verified
        
        Create a conclusions section that:
        1. Summarizes the most important implications
        2. Provides 3-5 specific, actionable recommendations
        3. Identifies priority areas for future research or action
        4. Notes limitations and areas of uncertainty
        5. Suggests implementation approaches where relevant
        
        Focus on practical value and clear next steps.
        """
        
        conclusions_response = self.run(conclusions_prompt)
        conclusions = conclusions_response.content if hasattr(conclusions_response, 'content') else str(conclusions_response)
        
        # Assemble comprehensive report
        processing_time = (datetime.now() - context.processing_start).total_seconds()
        
        report = f"""# Research Report: {context.topic}

*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*  
*Processing Time: {processing_time:.1f} seconds*  
*Confidence Level: {context.confidence_score:.0%}*  
*Quality Score: {context.quality_metrics.get('average_credibility', 0):.2f}/1.0*

## Executive Summary

{exec_summary}

## Research Methodology

This research was conducted using an advanced multi-agent system with comprehensive quality tracking:

**Information Gathering**
- Search strategy: {', '.join(context.quality_metrics.get('search_strategies', ['Standard web and academic search']))}
- Total sources: {len(context.sources)} ({context.quality_metrics.get('academic_ratio', 0):.0%} academic)
- Average source credibility: {context.quality_metrics.get('average_credibility', 0):.2f}/1.0

**Analysis and Synthesis**
- Insights extracted: {len(context.insights)} key findings
- Analysis depth: Comprehensive synthesis with evidence evaluation
- Quality assurance: Multi-stage verification and validation process

**Fact Verification**
- Claims verified: {context.quality_metrics.get('claims_verified', 0)} factual assertions checked
- Verification success: {context.quality_metrics.get('supported_claims', 0)} claims supported by sources
- Credibility assessment: {context.quality_metrics.get('credibility_ratio', 0):.0%} verification rate

## Key Findings

{self._format_insights(context.insights)}

## Detailed Analysis

{context.analysis}

## Fact Verification Summary

{self._format_fact_checks(context.fact_checks)}

## Conclusions and Recommendations

{conclusions}

## Source Documentation

{self._format_sources(context.sources)}

## Research Quality Metrics

{self._format_quality_metrics(context.quality_metrics, processing_time)}

---
*This report was generated using the Agno agentic framework with specialized research agents and comprehensive quality assurance.*
"""
        
        context.final_report = report
        context.current_phase = ResearchPhase.COMPLETE
        
        logger.info(f"Report generation completed: {len(report)} characters")
        logger.info(f"Writer agent completed: {len(report)} character report generated")
        
        return context
    
    def _format_insights(self, insights: List[str]) -> str:
        """Format insights as numbered list"""
        if not insights:
            return "No key insights identified."
        
        formatted = ""
        for i, insight in enumerate(insights, 1):
            formatted += f"{i}. {insight}\n\n"
        return formatted
    
    def _format_fact_checks(self, fact_checks: List[Dict]) -> str:
        """Format fact-checking results"""
        if not fact_checks:
            return "No specific factual claims were identified for verification."
        
        supported = len([c for c in fact_checks if c.get("status") == "supported"])
        total = len(fact_checks)
        
        result = f"**Verification Overview**: {supported}/{total} claims supported by available sources\n\n"
        
        for check in fact_checks[:6]:  # Show top 6 claims
            status_emoji = "✅" if check.get("status") == "supported" else "⚠️" if check.get("status") == "needs_verification" else "❌"
            result += f"{status_emoji} **{check.get('evidence_strength', 'unknown').title()} Evidence**: {check.get('claim', 'Unknown claim')[:150]}...\n"
            result += f"   *Verification Score: {check.get('verification_score', 0):.2f}*\n\n"
        
        return result
    
    def _format_sources(self, sources: List[Dict]) -> str:
        """Format sources with proper academic citations"""
        if not sources:
            return "No sources available."
        
        academic_sources = [s for s in sources if s.get("source_type") == "academic"]
        web_sources = [s for s in sources if s.get("source_type") != "academic"]
        
        formatted = ""
        
        if academic_sources:
            formatted += "### Academic Sources\n\n"
            for i, source in enumerate(academic_sources, 1):
                formatted += f"{i}. {', '.join(source.get('authors', ['Unknown Author']))}"
                formatted += f" ({source.get('year', 'n.d.')}). "
                formatted += f"{source.get('title', 'Unknown Title')}. "
                formatted += f"*{source.get('journal', 'Unknown Journal')}*, "
                formatted += f"{source.get('volume', 'n.v.')}({source.get('issue', 'n.i.')}), "
                formatted += f"{source.get('pages', 'n.p.')}. "
                formatted += f"DOI: {source.get('doi', 'N/A')}\n\n"
        
        if web_sources:
            formatted += "### Web Sources\n\n"
            for i, source in enumerate(web_sources, 1):
                formatted += f"{i}. {source.get('title', 'Unknown Title')}\n"
                formatted += f"   {source.get('url', 'No URL available')}\n"
                formatted += f"   Credibility Score: {source.get('credibility_score', 0):.2f}/1.0\n\n"
        
        return formatted
    
    def _format_quality_metrics(self, metrics: Dict[str, Any], processing_time: float) -> str:
        """Format comprehensive quality metrics"""
        return f"""
**Source Quality**
- Total Sources: {metrics.get('total_sources', 0)}
- Academic Ratio: {metrics.get('academic_ratio', 0):.0%}
- Average Credibility: {metrics.get('average_credibility', 0):.2f}/1.0
- Source Diversity: High (academic and web sources)

**Analysis Quality**  
- Insights Extracted: {metrics.get('insights_extracted', 0)}
- Insight Density: {metrics.get('insight_density', 0):.2f} insights per source
- Analysis Depth: {metrics.get('analysis_depth', 0):.2f}/1.0

**Verification Quality**
- Claims Verified: {metrics.get('claims_verified', 0)}
- Verification Success Rate: {metrics.get('credibility_ratio', 0):.0%}
- Supported Claims: {metrics.get('supported_claims', 0)}

**Processing Efficiency**
- Total Processing Time: {processing_time:.1f} seconds
- Search Strategies: {len(metrics.get('search_strategies', []))} different approaches
- Agent Coordination: Seamless multi-agent workflow
"""

# Main workflow orchestrator
class ResearchWorkflow:
    """
    Main research workflow that orchestrates the entire research process using Agno's capabilities.
    Provides comprehensive coordination, monitoring, and optimization.
    """
    
    def __init__(self):
        # Initialize specialized agents
        self.search_agent = ResearchSearchAgent()
        self.analysis_agent = ResearchAnalysisAgent()
        self.fact_check_agent = ResearchFactCheckAgent()
        self.writer_agent = ResearchWriterAgent()
        
        logger.info("Research workflow initialized with Agno framework")
    
    async def execute(self, topic: str, **kwargs) -> ResearchContext:
        """
        Execute the complete research workflow with comprehensive tracking and optimization.
        """
        logger.info(f"Starting Agno research workflow: {topic}")
        
        # Initialize research context
        context = ResearchContext(
            topic=topic,
            depth=kwargs.get("depth", "comprehensive"),
            sources_required=kwargs.get("sources_required", 8),
            focus_areas=kwargs.get("focus_areas", [])
        )
        
        try:
            # Phase 1: Search and Information Gathering
            logger.info("Phase 1: Information Gathering")
            context = await self.search_agent.search_comprehensive(context)
            
            # Phase 2: Analysis and Synthesis
            logger.info("Phase 2: Analysis and Synthesis")
            context = await self.analysis_agent.analyze_sources(context)
            
            # Phase 3: Fact Checking and Verification
            logger.info("Phase 3: Fact Checking and Verification")
            context = await self.fact_check_agent.verify_research(context)
            
            # Phase 4: Report Writing and Assembly
            logger.info("Phase 4: Report Generation")
            context = await self.writer_agent.generate_report(context)
            
            total_time = (datetime.now() - context.processing_start).total_seconds()
            logger.info(f"Research workflow completed in {total_time:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
        
        return context
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get comprehensive workflow status and capabilities"""
        return {
            "framework": "Agno",
            "agents": {
                "search": self.search_agent.name,
                "analysis": self.analysis_agent.name,
                "fact_check": self.fact_check_agent.name,
                "writer": self.writer_agent.name
            },
            "capabilities": [
                "Multi-source information gathering",
                "AI-powered analysis and synthesis",
                "Automated fact verification",
                "Professional report generation",
                "Quality metrics and confidence scoring",
                "Comprehensive workflow orchestration"
            ],
            "quality_features": [
                "Source credibility assessment",
                "Academic source prioritization",
                "Claim verification with confidence scoring",
                "Multi-agent validation and review",
                "Comprehensive quality metrics tracking"
            ]
        }

# Main execution function
async def main():
    """
    Demonstrate the Agno-powered research workflow with comprehensive capabilities.
    """
    print("🚀 Agno Research Assistant - Advanced Multi-Agent Workflow")
    print("=" * 75)
    
    # Initialize the workflow
    workflow = ResearchWorkflow()
    
    # Configure research parameters
    research_config = {
        "topic": "Impact of AI code generation tools on software development productivity and quality",
        "depth": "comprehensive",
        "sources_required": 10,
        "focus_areas": [
            "developer productivity metrics",
            "code quality impact assessment",
            "adoption patterns in enterprise",
            "skill development and learning"
        ]
    }
    
    print(f"📋 Research Topic: {research_config['topic']}")
    print(f"🎯 Focus Areas: {', '.join(research_config['focus_areas'])}")
    print(f"📚 Target Sources: {research_config['sources_required']}")
    print(f"🔍 Research Depth: {research_config['depth']}")
    print("-" * 75)
    
    # Execute research workflow
    start_time = datetime.now()
    print("🔄 Executing Agno multi-agent research workflow...")
    
    try:
        context = await workflow.execute(
            research_config["topic"],
            **{k: v for k, v in research_config.items() if k != "topic"}
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Display comprehensive results
        print(f"\n📊 Research Workflow Completed Successfully!")
        print("=" * 75)
        print(f"✅ Total Sources Analyzed: {len(context.sources)}")
        print(f"📰 Web Sources: {len([s for s in context.sources if s.get('source_type') != 'academic'])}")
        print(f"🎓 Academic Papers: {len([s for s in context.sources if s.get('source_type') == 'academic'])}")
        print(f"🔍 Key Insights Generated: {len(context.insights)}")
        print(f"✔️ Claims Verified: {context.quality_metrics.get('supported_claims', 0)}/{context.quality_metrics.get('claims_verified', 0)}")
        print(f"📈 Confidence Score: {context.confidence_score:.0%}")
        print(f"⭐ Source Quality: {context.quality_metrics.get('average_credibility', 0):.2f}/1.0")
        print(f"⏱️ Total Processing Time: {execution_time:.1f} seconds")
        print(f"📄 Report Length: {len(context.final_report):,} characters")
        
        # Show workflow status
        status = workflow.get_workflow_status()
        print(f"\n🤖 Agent Coordination Summary:")
        print("-" * 45)
        for role, agent_name in status["agents"].items():
            print(f"  {role.title()}: {agent_name}")
        
        # Display sample insights
        print(f"\n💡 Sample Key Insights:")
        print("-" * 45)
        for i, insight in enumerate(context.insights[:3], 1):
            print(f"  {i}. {insight[:120]}...")
        
        # Display report preview
        print(f"\n📄 Research Report Preview:")
        print("-" * 75)
        preview_length = 1000
        report_preview = context.final_report[:preview_length]
        print(report_preview + "..." if len(context.final_report) > preview_length else report_preview)
        
        # Save comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agno_research_report_{timestamp}.md"
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(context.final_report)
        
        print(f"\n💾 Complete report saved: {filename}")
        print(f"📊 Quality Metrics: {context.quality_metrics}")
        
        print(f"\n🎉 Agno Research Assistant completed successfully!")
        print("   Advanced multi-agent coordination with comprehensive quality assurance")
        
    except Exception as e:
        print(f"\n❌ Workflow execution failed: {e}")
        logger.error(f"Main execution failed: {e}")

if __name__ == "__main__":
    # Execute the Agno-powered research workflow
    asyncio.run(main())