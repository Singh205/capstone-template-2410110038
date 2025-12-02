

from typing import TypedDict, List, Annotated, Optional
import operator
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
from datetime import datetime
import os


load_dotenv()



class Loophole(BaseModel):
    """Model for contract loopholes"""
    clause_reference: str = Field(description="Reference to the specific clause")
    loophole_description: str = Field(description="Detailed description of the loophole")
    severity: str = Field(description="Severity level: HIGH, MEDIUM, or LOW")
    exploitation_scenario: str = Field(description="How this could be exploited")

class RiskItem(BaseModel):
    """Model for risk assessment"""
    risk_type: str = Field(description="Category of risk")
    description: str = Field(description="Detailed risk description")
    severity_score: int = Field(description="Severity score from 1-10", ge=1, le=10)
    affected_party: str = Field(description="Which party is affected")
    mitigation: Optional[str] = Field(default=None, description="Suggested mitigation strategy")

class ContractAnalysis(BaseModel):
    """Complete contract analysis output"""
    overall_risk_score: int = Field(description="Overall risk score 1-100", ge=1, le=100)
    summary: str = Field(description="Executive summary of the contract analysis")
    loopholes: List[Loophole] = Field(description="List of identified loopholes")
    risks: List[RiskItem] = Field(description="List of identified risks")
    red_flags: List[str] = Field(description="List of immediate concerns")
    recommendations: List[str] = Field(description="Actionable recommendations")



class ContractState(TypedDict):
    """State object that flows through the graph"""
    file_path: str
    raw_text: str
    chunks: List[str]
    vectorstore: Optional[object]
    search_queries: List[str]
    retrieved_context: List[str]
    analysis: Optional[ContractAnalysis]
    error: Optional[str]
    metadata: dict
    processing_step: str



class ContractAnalyzerGraph:
    """LangGraph-based Contract Analyzer"""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        """Initialize the analyzer with OpenAI models"""
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        
        # Build the workflow graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ContractState)
        
        # Add all nodes in the pipeline
        workflow.add_node("load_document", self.load_document)
        workflow.add_node("create_chunks", self.create_chunks)
        workflow.add_node("build_vectorstore", self.build_vectorstore)
        workflow.add_node("identify_queries", self.identify_search_queries)
        workflow.add_node("retrieve_context", self.retrieve_context)
        workflow.add_node("analyze_contract", self.analyze_contract)
        workflow.add_node("generate_report", self.generate_report)
        
        # Define the workflow edges
        workflow.set_entry_point("load_document")
        workflow.add_edge("load_document", "create_chunks")
        workflow.add_edge("create_chunks", "build_vectorstore")
        workflow.add_edge("build_vectorstore", "identify_queries")
        workflow.add_edge("identify_queries", "retrieve_context")
        workflow.add_edge("retrieve_context", "analyze_contract")
        workflow.add_edge("analyze_contract", "generate_report")
        workflow.add_edge("generate_report", END)
        
        return workflow.compile()
    
    
    
    def load_document(self, state: ContractState) -> ContractState:
        """Node 1: Load document from file path"""
        state["processing_step"] = "Loading document"
        print(f"\nLoading document: {state['file_path']}")
        
        try:
            file_path = state["file_path"]
            
            # Determine loader based on file extension
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith('.txt'):
                loader = TextLoader(file_path)
            else:
                raise ValueError("Unsupported file format. Use PDF or TXT files.")
            
            # Load document
            documents = loader.load()
            raw_text = "\n\n".join([doc.page_content for doc in documents])
            
            # Update state
            state["raw_text"] = raw_text
            state["metadata"] = {
                "file_name": os.path.basename(file_path),
                "pages": len(documents),
                "timestamp": datetime.now().isoformat(),
                "word_count": len(raw_text.split())
            }
            
            
            
        except Exception as e:
            state["error"] = f"Document loading error: {str(e)}"
            print(f"   ✗ Error: {e}")
        
        return state
    
    def create_chunks(self, state: ContractState) -> ContractState:
        """Node 2: Split document into chunks for processing"""
        state["processing_step"] = "Creating chunks"
        
        
        if state.get("error"):
            return state
        
        try:
            chunks = self.text_splitter.split_text(state["raw_text"])
            state["chunks"] = chunks
            print(f"   ✓ Created {len(chunks)} chunks")
        except Exception as e:
            state["error"] = f"Chunking error: {str(e)}"
            print(f"   ✗ Error: {e}")
        
        return state
    
    def build_vectorstore(self, state: ContractState) -> ContractState:
        """Node 3: Build vector store for semantic search"""
        state["processing_step"] = "Building vector store"
        
        
        if state.get("error"):
            return state
        
        try:
            from langchain_core.documents import Document
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
            
            
            docs = [Document(page_content=chunk) for chunk in state["chunks"]]
            
            
            vectorstore = FAISS.from_documents(
                documents=docs,
                embedding=self.embeddings
            )
            
            state["vectorstore"] = vectorstore
            
            
        except Exception as e:
            state["error"] = f"Vector store error: {str(e)}"
            print(f"   ✗ Error: {e}")
        
        return state
    
    def identify_search_queries(self, state: ContractState) -> ContractState:
        """Node 4: Use LLM to identify key areas to investigate"""
        state["processing_step"] = "Identifying investigation areas"
        print("\nIdentifying areas to investigate...")
        
        if state.get("error"):
            return state
        
        try:
           
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a legal expert identifying key investigation areas in contracts."),
                ("user", """Analyze this contract excerpt and identify 6 specific areas that need 
detailed investigation for potential risks, loopholes, and unfavorable terms.

Contract excerpt (first 2000 characters):
{contract_excerpt}

Return ONLY a JSON object in this exact format:
{{"queries": ["specific query 1", "specific query 2", "specific query 3", "specific query 4", "specific query 5", "specific query 6"]}}

Focus on:
- Payment and financial terms
- Liability and indemnification
- Termination conditions
- Data privacy and security
- Intellectual property rights
- Dispute resolution
""")
            ])
            
            chain = prompt | self.llm
            response = chain.invoke({
                "contract_excerpt": state['raw_text'][:2000]
            })
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            
            if json_match:
                queries_data = json.loads(json_match.group())
                queries = queries_data.get("queries", [])
            else:
                # Fallback queries if parsing fails
                queries = [
                    "payment terms, late fees, and financial penalties",
                    "liability limitations and indemnification clauses",
                    "termination rights and exit conditions",
                    "data usage, privacy, and confidentiality provisions",
                    "intellectual property ownership and licensing",
                    "dispute resolution, arbitration, and jurisdiction"
                ]
            
            state["search_queries"] = queries
            print(f"   ✓ Identified {len(queries)} investigation areas")
            for i, q in enumerate(queries, 1):
                print(f"      {i}. {q[:60]}...")
            
        except Exception as e:
            state["error"] = f"Query identification error: {str(e)}"
            
        
        return state
    
    def retrieve_context(self, state: ContractState) -> ContractState:
        """Node 5: Retrieve relevant context using RAG"""
        state["processing_step"] = "Retrieving context"
        print("\nRetrieving relevant context...")
        
        if state.get("error"):
            return state
        
        try:
            vectorstore = state["vectorstore"]
            retrieved = []
            
            for query in state["search_queries"]:
                # Semantic search for each query
                results = vectorstore.similarity_search(query, k=3)
                context = "\n---\n".join([doc.page_content for doc in results])
                retrieved.append(f"Investigation Area: {query}\n\nRelevant Context:\n{context}")
                print(f"   ✓ Retrieved context for: {query[:50]}...")
            
            state["retrieved_context"] = retrieved
            
            
        except Exception as e:
            state["error"] = f"Context retrieval error: {str(e)}"
            
        
        return state
    
    def analyze_contract(self, state: ContractState) -> ContractState:
        """Node 6: Perform comprehensive contract analysis"""
        state["processing_step"] = "Analyzing contract"
        print("\nAnalyzing contract with structured output...")
        
        if state.get("error"):
            return state
        
        try:
            
            analysis_llm = self.llm.with_structured_output(ContractAnalysis)
            
            analysis_prompt = f"""You are an expert contract attorney with 20 years of experience.
Perform a comprehensive analysis of this contract.

FULL CONTRACT TEXT:
{state['raw_text']}

RETRIEVED CONTEXT FROM TARGETED INVESTIGATION:
{chr(10).join(state['retrieved_context'])}

Provide a detailed analysis including:

1. Overall Risk Score (1-100): Consider all factors
2. Executive Summary: Brief overview of key findings
3. Loopholes: Specific clauses with potential for exploitation
4. Risk Assessment: Categorized risks with severity scores
5. Red Flags: Immediate concerns that need attention
6. Recommendations: Actionable steps for negotiation

Be thorough, specific, and reference actual clauses when possible.
"""
            
            analysis = analysis_llm.invoke(analysis_prompt)
            state["analysis"] = analysis
            
            print(f"   ✓ Analysis complete")
            print(f"   ✓ Risk Score: {analysis.overall_risk_score}/100")
            print(f"   ✓ Loopholes found: {len(analysis.loopholes)}")
            print(f"   ✓ Risks identified: {len(analysis.risks)}")
            
        except Exception as e:
            state["error"] = f"Analysis error: {str(e)}"
            print(f"   ✗ Error: {e}")
        
        return state
    
    def generate_report(self, state: ContractState) -> ContractState:
        """Node 7: Generate and save final report"""
        state["processing_step"] = "Generating report"
        print("\nGenerating comprehensive report...")
        
        if state.get("error"):
            self._print_error_report(state)
            return state
        
        analysis = state["analysis"]
        
        # Print detailed report
        self._print_analysis_report(state, analysis)
        
        # Save JSON report
        self._save_json_report(state, analysis)
        
        return state
    
    
    
    def _print_analysis_report(self, state: ContractState, analysis: ContractAnalysis):
        """Print formatted analysis report to console"""
        print(f"\n{'='*80}")
        print(f"{'CONTRACT RISK ANALYSIS REPORT':^80}")
        print(f"{'='*80}")
        print(f"\n📄 File: {state['metadata']['file_name']}")
        print(f"📊 Pages: {state['metadata']['pages']}")
        print(f"📝 Word Count: {state['metadata']['word_count']}")
        print(f"📅 Analysis Date: {state['metadata']['timestamp']}")
        print(f"\n{'='*80}")
        
        # Risk Score
        risk_score = analysis.overall_risk_score
        if risk_score >= 70:
            risk_level = "🔴 HIGH RISK - SIGNIFICANT CONCERNS"
        elif risk_score >= 40:
            risk_level = "🟡 MEDIUM RISK - CAUTION ADVISED"
        else:
            risk_level = "🟢 LOW RISK - ACCEPTABLE TERMS"
        
        print(f"\n🎯 OVERALL RISK SCORE: {risk_score}/100")
        print(f"   {risk_level}")
        
        # Summary
        print(f"\n{'─'*80}")
        print(f"📋 EXECUTIVE SUMMARY")
        print(f"{'─'*80}")
        print(f"\n{analysis.summary}\n")
        
        # Loopholes
        print(f"{'─'*80}")
        print(f"🔓 LOOPHOLES IDENTIFIED: {len(analysis.loopholes)}")
        print(f"{'─'*80}")
        for i, loophole in enumerate(analysis.loopholes, 1):
            severity_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(
                loophole.severity.upper(), "⚪"
            )
            print(f"\n{i}. {severity_emoji} [{loophole.severity}] {loophole.clause_reference}")
            print(f"   📝 Description: {loophole.loophole_description}")
            print(f"   ⚠️  Exploitation Risk: {loophole.exploitation_scenario}")
        
        # Risks
        print(f"\n{'─'*80}")
        print(f"⚠️  DETAILED RISK ASSESSMENT: {len(analysis.risks)}")
        print(f"{'─'*80}")
        for i, risk in enumerate(analysis.risks, 1):
            score_emoji = "🔴" if risk.severity_score >= 7 else "🟡" if risk.severity_score >= 4 else "🟢"
            print(f"\n{i}. {score_emoji} {risk.risk_type.upper()} - Severity: {risk.severity_score}/10")
            print(f"   👥 Affected Party: {risk.affected_party}")
            print(f"   📝 {risk.description}")
            if risk.mitigation:
                print(f"   💡 Mitigation: {risk.mitigation}")
        
        # Red Flags
        print(f"\n{'─'*80}")
        print(f"🚩 RED FLAGS: {len(analysis.red_flags)}")
        print(f"{'─'*80}")
        for i, flag in enumerate(analysis.red_flags, 1):
            print(f"\n{i}. {flag}")
        
        # Recommendations
        print(f"\n{'─'*80}")
        print(f"💡 RECOMMENDATIONS: {len(analysis.recommendations)}")
        print(f"{'─'*80}")
        for i, rec in enumerate(analysis.recommendations, 1):
            print(f"\n{i}. {rec}")
        
        print(f"\n{'='*80}\n")
    
    def _save_json_report(self, state: ContractState, analysis: ContractAnalysis):
        """Save analysis report as JSON file"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs("analysis_reports", exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = f"analysis_reports/contract_analysis_{timestamp}.json"
            
            # Prepare report data
            report_data = {
                "metadata": state["metadata"],
                "risk_score": analysis.overall_risk_score,
                "analysis": analysis.model_dump()
            }
            
            # Save to file
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Report saved: {report_file}")
            
        except Exception as e:
            print(f"⚠️  Could not save JSON report: {e}")
    
    def _print_error_report(self, state: ContractState):
        """Print error report if workflow failed"""
        print(f"\n{'='*80}")
        print(f"{'ERROR REPORT':^80}")
        print(f"{'='*80}")
        print(f"\n❌ Workflow failed at step: {state.get('processing_step', 'Unknown')}")
        print(f"❌ Error: {state.get('error', 'Unknown error')}")
        print(f"\n{'='*80}\n")
    
    
    
    def analyze_file(self, file_path: str) -> ContractState:
        """
        Analyze a contract file through the complete LangGraph workflow
        
        Args:
            file_path: Path to PDF or TXT contract file
            
        Returns:
            Final state with complete analysis
        """
        # Initialize state
        initial_state = {
            "file_path": file_path,
            "raw_text": "",
            "chunks": [],
            "vectorstore": None,
            "search_queries": [],
            "retrieved_context": [],
            "analysis": None,
            "error": None,
            "metadata": {},
            "processing_step": "Initialization"
        }
        
        print(f"\n{'='*80}")
        
        print(f"{'='*80}")
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        print(f"\n{'='*80}")
        if final_state.get("error"):
            print(f"{'WORKFLOW FAILED':^80}")
        else:
            print(f"{'✅ WORKFLOW COMPLETE':^80}")
        print(f"{'='*80}\n")
        
        return final_state



def main():
       
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in a .env file or environment")
        return
    
    # Initialize analyzer
    print("\n🔧 Initializing Contract Analyzer...")
    analyzer = ContractAnalyzerGraph(model_name="gpt-4o-mini")
    print("✓ Analyzer ready\n")
    
    
    contract_path = "test_contracts\SampleContract-Shuttle.pdf"
    
    if os.path.exists(contract_path):
        result = analyzer.analyze_file(contract_path)
        
        if not result.get("error"):
            print("\n Analysis completed successfully!")
            print(f"Risk Score: {result['analysis'].overall_risk_score}/100")
        else:
            print(f"\n❌ Analysis failed: {result['error']}")
    else:
        print(f"\nSample contract not found at: {contract_path}")
        print("Please provide a valid contract file path")
        print("\nExample usage:")
        print("  analyzer = ContractAnalyzerGraph()")
        print("  result = analyzer.analyze_file('path/to/your/contract.pdf')")


if __name__ == "__main__":
    main()