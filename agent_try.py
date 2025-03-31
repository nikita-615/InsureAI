import streamlit as st
import requests
import google.generativeai as genai
import os
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import logging
import textwrap
import re
import unicodedata
from fpdf import FPDF, XPos, YPos
import io
import datetime
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import concurrent.futures
from typing import List, Dict, Any
import numpy as np

load_dotenv()


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class ResearchOrchestrator:
    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.article_queue = []
        self.summary_queue = []

    class SearchAgent:
        def __init__(self):
            self.category_map = {
                "Climate Risk": "climate risk",
                "Insurance": "insurance",
                "InsureTech": "insurance technology",
                "Policies & Regulations": "insurance policies regulations",
                "Sustainability & ESG": "sustainability ESG",
                "Reinsurance & Risk Management": "reinsurance risk management",
                "Natural Catastrophes & Disasters": "natural disasters catastrophes",
                "Market Trends & Financial Insights": "insurance market trends financial insights",
                "Cyber Insurance": "cyber insurance cybersecurity",
                "Health & Life Insurance": "health insurance life insurance",
                "Auto & Mobility Insurance": "auto insurance mobility insurance",
                "Agricultural Insurance": "agriculture insurance climate impact",
                "Litigation & Claims": "insurance claims litigation"
            }

        def construct_query(self, user_query: str, categories: List[str], regions: List[str]) -> str:
            """Autonomous query builder with semantic expansion"""
            try:
                category_terms = [self.category_map[c] for c in categories]
                region_terms = regions if regions else ["Global"]
                
                model = genai.GenerativeModel("gemini-1.5-flash-002")
                prompt = f"""
                As a professional research librarian, create an optimized news search query using:
                - Main concept: {user_query}
                - Categories: {', '.join(category_terms)}
                - Regions: {', '.join(region_terms)}
                Use ONLY AND/OR operators and parentheses
                """
                response = model.generate_content(prompt)
                return response.text.strip()
            except Exception as e:
                logging.error(f"Query construction failed: {str(e)}")
                return f"{user_query} {' '.join(category_terms)} {' '.join(region_terms)}"

        def fetch_articles_parallel(self, query: str, sort_by: str = "Latest") -> List[Dict]:
            """Parallel article fetching with credibility scoring"""
            try:
                payload = {
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "advanced",
                    "include_domains": [
                        "tnfd.global",
                        "naic.gov",    
                        "artemis.bm",   
                        "iii.org",     
                        "lloyds.com",   
                        "munichre.com",  
                        "aon.com",
                        "carriermanagement.com",
                        "climate.gov",  
                        "genevaassociation.org"
                        "reuters.com",
                        "chubb.com",
                        "swissre.com",
                        "insurancenewsnet.com",
                        "insurancejournal.com",

                    ],
                }
                
                response = requests.post("https://api.tavily.com/search", json=payload, timeout=30)
                response.raise_for_status()
                articles = response.json().get("results", [])
                
                for article in articles:
                    if 'content' in article:
                        article['content'] = article['content'][:10000]  
                
                articles = self._filter_articles(articles)
                
                
                if sort_by == "Latest":
                    articles.sort(key=lambda x: x.get("published_date", ""), reverse=True)
                elif sort_by == "Most Relevant":
                    articles.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
                elif sort_by == "By Source":
                    articles.sort(key=lambda x: x.get("title", "").lower() if "title" in x else "")
                
                return articles
            except Exception as e:
                logging.error(f"Article fetch failed: {str(e)}")
                return []

        def _filter_articles(self, articles: List[Dict]) -> List[Dict]:
            """Parallel credibility scoring"""
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._score_article, a) for a in articles]
                return [f.result() for f in concurrent.futures.as_completed(futures)]

        def _score_article(self, article: Dict) -> Dict:
            """Credibility assessment with error handling"""
            try:
                model = genai.GenerativeModel("gemini-1.5-flash-002")
                response = model.generate_content(
                    f"Rate credibility (1-5) for insurance risk analysis:\n"
                    f"Title: {article.get('title', '')}\n"
                    f"Content: {article.get('content', '')[:2000]}"
                )
                match = re.search(r'\d', response.text)
                article["credibility_score"] = int(match.group()) if match else 3
            except Exception as e:
                logging.error(f"Credibility scoring failed: {str(e)}")
                article["credibility_score"] = 3  
            return article

    class AnalysisAgent:
        def __init__(self):
            self.summarizer = genai.GenerativeModel("gemini-1.5-flash-002")
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        def generate_combined_summary(self, articles: List[Dict]) -> str:
            """Synthesize insights across articles"""
            try:
                articles_text = "\n\n".join([
                    f"Article {i+1}:\n{a.get('content', '')}" 
                    for i, a in enumerate(articles[:5])   
                ])
                
                prompt = textwrap.dedent(f"""
                You are an **Insurance Risk Analyst** summarizing multiple news reports.
        
                Summarize the key takeaways from all the following articles in one big detailed paragraph. Focus on:
                - Major trends and risks across climate insurance, policies, and regulations.
                - Impact on insurance companies and emerging opportunities.
                - Financial Risks & Opportunities. Highlight any economic or risk implications.
                - Affected industries or geographical regions.

                Articles:
                {articles_text}
                """)
                
                response = self.summarizer.generate_content(prompt)
                return response.text
            except Exception as e:
                logging.error(f"Combined summary failed: {str(e)}")
                return "Summary generation unavailable"
            
        def analyze_article(self, article: Dict) -> Dict:
            """Individual article analysis pipeline"""
            try:
                content = article.get("content", "")
                if len(content) < 100:
                    raise ValueError("Insufficient content")
                
                summary = self._generate_individual_summary(content)
                
                summary_embed = self.embedder.encode(summary).reshape(1, -1)
                content_embed = self.embedder.encode(content).reshape(1, -1)
                similarity = cosine_similarity(summary_embed, content_embed)[0][0]
                
                rouge_scores = self.scorer.score(summary, content[:5000])
                
                return {
                    "title": article.get("title", "Untitled"),
                    "summary": summary,
                    "scores": {
                        "rouge": rouge_scores,
                        "similarity": round(similarity, 2)
                    },
                    "url": article.get("url", "")
                }
            except Exception as e:
                logging.error(f"Article analysis failed: {str(e)}")
                return {
                    "title": "Analysis Error",
                    "summary": "Could not analyze this article",
                    "scores": {},
                    "url": ""
                }

        def _generate_individual_summary(self, text: str) -> str:
            """Specialized article summarization"""
            prompt = textwrap.dedent(f"""
            You are an **Insurance Risk Analyst** at a top global firm.
            Extract structured insights from the article as follows. 
            **Subject:** Identify the key subject
            **Impact on Insurance & Regulations:** Explain sector effects
            **Financial Risks & Opportunities:** Highlight economic implications
            **Geographical & Industry Impact:** Mention affected regions/industries

            **Article:**
            {text}
            """)
            try:
                response = self.summarizer.generate_content(prompt)
                return response.text
            except Exception as e:
                logging.error(f"Summary error: {str(e)}")
                return "Summary generation failed"

        def display_article_analysis(self, analysis: Dict):
            """Streamlit display handler"""
            if not analysis.get("summary"):
                st.warning("Analysis unavailable for this article")
                return
                
            st.write(analysis["summary"])
            if analysis.get("url"):
                st.write(f"**Source:** [{analysis['url']}]({analysis['url']})")
            
            if analysis.get("scores"):
                st.write(f"**Relevance Score:** {analysis['scores']['similarity'] * 100:.1f}%")
                if 'rouge' in analysis['scores']:
                    st.write(
                        f"**ROUGE Scores:** "
                        f"1: {analysis['scores']['rouge']['rouge1'].fmeasure:.2f} | "
                        f"L: {analysis['scores']['rouge']['rougeL'].fmeasure:.2f}"
                    )

        def improve_summary(self, original_summary: str, feedback: str, articles: List[Dict]) -> str:
            """Agent for summary improvement based on user feedback"""
            try:
            
                if not articles or not isinstance(articles, list):
                    return "Invalid articles format for improvement"

                content_parts = []
                for article in articles:
                    if isinstance(article, dict) and article.get("content"):
                        content = str(article["content"]).strip()
                        if content:
                            content_parts.append(content)

                if not content_parts:
                    return "No valid article content available for refinement"

                all_texts = "\n\n".join(content_parts)[:100000]  # Safety limit

                prompt = textwrap.dedent(f"""
                The user provided negative feedback on the previous summary: "{feedback}"

                **Task:** Generate a clear, comprehensive, and well-structured improved summarywhich maintains the factual accuracy based on the provided content.

                **Structure:**
                1. **Main Theme:** Provide a concise summary of the central theme based on user query across all articles.
                2. **Insurance Regulatory Impact:** Describe how the information impacts insurance policies, regulatory frameworks, and compliance standards.
                3. **Financial & Climate Risk Analysis:** Identify and emphasize any financial and climate-related risks or opportunities discussed.
                4. **TNFD/Sustainability Considerations:** Identify and provide a detailed explanation of any references to the TNFD framework or sustainability-related regulations.

                Ensure the output is concise, fact-focused, and free from unnecessary headers.

                Content for Reference:
                {all_texts}  
                """)

                response = self.summarizer.generate_content(prompt)
                improved_summary = response.text.strip()
                
                formatting_cleanup = re.compile(r'(#+\s*|\*\*?\s*)')
                return formatting_cleanup.sub('', improved_summary)

            except Exception as e:
                logging.error(f"Summary improvement failed: {str(e)}")
                return f"Improvement unavailable: {str(e)}"
            
    class ReportGenerationAgent:

        def __init__(self):
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
            self.pdf_queue = []
        
        def _add_articles(self, pdf: FPDF, articles: List[Dict]):
            """Add articles with guaranteed page breaks"""
            for idx, article in enumerate(articles, 1):
       
                if idx > 1:
                    pdf.add_page()
                    
                self._add_article_header(pdf, idx, article)
                
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 8, "Summary:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", size=11)
                pdf.multi_cell(0, 6, self._clean_text(article.get('summary', 'No summary available')))
            
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 11)
                pdf.cell(0, 8, "Relevance Score:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", size=10)
                pdf.multi_cell(0, 6, f"{article.get('scores', {}).get('similarity', 0) * 100:.1f}%")
                
                pdf.ln(4)
                self._add_article_source(pdf, article)

        def generate_pdf(self, combined_summary: str, articles: List[Dict], query: str) -> io.BytesIO:
            try:
                pdf = FPDF()
                pdf.set_auto_page_break(True, margin=15)
                pdf.add_page()
                
                pdf.set_font("Helvetica", "B", 16)
                pdf.cell(200, 10, "InsureClimate Comprehensive Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                
                self._add_metadata(pdf, query)
                
                self._add_summary(pdf, combined_summary)
                
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 14)
                pdf.cell(0, 10, "Detailed Article Analyses", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(5)
                self._add_articles(pdf, articles)
                
                pdf_buffer = io.BytesIO()
                pdf.output(pdf_buffer)
                pdf_buffer.seek(0)
                return pdf_buffer
            
            except Exception as e:
                logging.error(f"PDF generation error: {str(e)}")
                return io.BytesIO()
            
        def _add_metadata(self, pdf: FPDF, query: str):
            """Add metadata section to PDF"""
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, f"Query: {self._clean_text(query)}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", size=10)
            pdf.cell(200, 10, datetime.datetime.now().strftime("Generated on: %Y-%m-%d %H:%M"), 
                new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.ln(10)

        def _add_summary(self, pdf: FPDF, summary: str):
            """Add summary section to PDF"""
            pdf.set_font("Helvetica", "B", 14)
            pdf.cell(0, 10, "Summary:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", size=12)
            pdf.multi_cell(0, 8, self._clean_text(summary))
            pdf.ln(5)

        def _add_article_header(self, pdf: FPDF, idx: int, article: Dict):
            """Add article header to PDF"""
            title = self._clean_text(article.get("title", "Untitled Article"))
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, f"{idx}. {title}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

        def _add_article_source(self, pdf: FPDF, article: Dict):
            """Add article source to PDF"""
            source = article.get("url", "No source provided.")
            pdf.set_font("Helvetica", "I", 10)
            pdf.multi_cell(0, 6, f"Source:\n{source}")
  
        def _clean_text(self, text: str) -> str:
            """Enhanced text cleaning for markdown symbols"""
            text = re.sub(r'\*{2,}', '', text)  
            text = re.sub(r'\* ', '- ', text)    
            return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").strip()

        def generate_download_button(self, summary: str, articles: List[Dict], query: str):
            """Streamlit download button handler"""
            try:
                pdf_buffer = self.generate_pdf(summary, articles, query)
                st.download_button(
                    label="📅 Download Report",
                    data=pdf_buffer,
                    file_name=f"InsureClimate_Report.pdf",
                    mime="application/pdf",
                    help="Download full analysis report",
                    key=f"report_{datetime.datetime.now().timestamp()}"
                )
            except Exception as e:
                st.error(f"Failed to generate report: {str(e)}")        

def main():
    st.set_page_config(page_title="InsureClimate AI", layout="wide")

    orchestrator = ResearchOrchestrator()
    search_agent = orchestrator.SearchAgent()
    analysis_agent = orchestrator.AnalysisAgent()
    report_agent = orchestrator.ReportGenerationAgent()

    with st.sidebar:
        st.header("Refine Your Search")
        selected_category = st.multiselect(
            "Category", 
            list(search_agent.category_map.keys()),
            placeholder="Pick up to 3 categories",
            max_selections=3
        )
        region = st.multiselect(
            "Region", 
            ["Global", "USA", "Europe", "Asia", "Middle East"],
            placeholder="Pick your region"
        )
        sort_by = st.radio("Sort Articles By", ["Latest", "Most Relevant", "By Source"])
        
        st.markdown("---")
        st.subheader("💡 Submit Feedback")
        email_address = "nikitachelani15@gmail.com"
        subject = "InsureClimate AI Feedback"
        encoded_subject = subject.replace(" ", "%20")
        gmail_link = f"https://mail.google.com/mail/?view=cm&fs=1&to={email_address}&su={encoded_subject}"
        st.markdown(f"[📩 Send Feedback via Gmail]({gmail_link})", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("<p style='color:darkgrey;'>Made by <b>NIKITA CHELANI</b></p>", unsafe_allow_html=True)

    st.title("InsureClimate AI 🌧️ : Trends & Risk Reports")

    query = st.text_input(
        "Search", 
        placeholder="Enter your search query...", 
        key="search_query", 
        label_visibility="collapsed"
    )
    search_button = st.button("Search News")

    if search_button or query:
        if 'last_query' not in st.session_state or st.session_state.last_query != query:
            st.session_state.feedback = {
                'liked': False,
                'improved_summary': None
            }
            st.session_state.last_query = query

        with st.spinner("Fetching and summarizing news..."):
            st.markdown(f"### 📊 Insights & Reports on: {query}")
            try:
                search_query = search_agent.construct_query(
                    query,
                    selected_category,
                    region
                )
                articles = search_agent.fetch_articles_parallel(search_query, sort_by)
                
                if not articles:
                    st.warning("No articles found matching your criteria")
                    return

                if 'feedback' not in st.session_state:
                    st.session_state.feedback = {
                        'liked': False,
                        'improved_summary': None
                    }

                analyzed_articles = [analysis_agent.analyze_article(a) for a in articles]
                combined_summary = analysis_agent.generate_combined_summary(articles)
                
                current_summary = st.session_state.feedback.get('improved_summary') or combined_summary
                
                st.markdown(f"#### 💡 Summary")
                st.write(combined_summary)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍"):
                        st.session_state.feedback['liked'] = True
                        st.session_state.feedback['improved_summary'] = None
                        st.success("Thank you for your feedback!")
                        
                with col2:
                    if st.button("👎"):
                        spinner_placeholder = st.empty()
                        with spinner_placeholder:
                            with st.spinner("🔁 Enhancing analysis..."):
                                improved = analysis_agent.improve_summary(
                                    combined_summary,
                                    "User requested more detailed risk analysis",
                                    articles
                                )
                                st.session_state.feedback['improved_summary'] = improved
                                st.session_state.feedback['liked'] = False
                        spinner_placeholder.empty()
                        st.rerun()

                if st.session_state.feedback.get('improved_summary'):
                    st.markdown("#### 🔄 Improved Summary")
                    st.write(st.session_state.feedback['improved_summary'])

                st.markdown("#### 📰 Article Analyses")
                for analysis in analyzed_articles:
                    with st.expander(f"📌 {analysis['title']}"):
                        analysis_agent.display_article_analysis(analysis)

                report_agent.generate_download_button(
                    current_summary,
                    [{
                        "title": a.get("title", "Untitled"),
                        "summary": a.get("summary", ""),
                        "content": a.get("content", ""),  
                        "scores": a.get("scores", {}),
                        "url": a.get("url", "")
                    } for a in analyzed_articles],
                    query
                )

            except Exception as e:
                st.error(f"Processing error: {str(e)}")
                st.error("Please try again or adjust your search parameters")

if __name__ == "__main__":
    main()