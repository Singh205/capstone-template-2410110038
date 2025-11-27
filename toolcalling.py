# Day 3: RAG Integration & Tool Calling
# Combines Day 1 & Day 2 with RAG

from dotenv import load_dotenv
from typing import List, Optional

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool  
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel, Field

load_dotenv()



class Loophole(BaseModel):
    clause_reference: str
    loophole_description: str
    severity: str
    exploitation_scenario: str

class RiskItem(BaseModel):
    risk_type: str
    description: str
    severity_score: int
    affected_party: str
    mitigation: Optional[str] = None

class ContractAnalysis(BaseModel):
    overall_risk_score: int
    summary: str
    loopholes: List[Loophole]
    risks: List[RiskItem]
    red_flags: List[str]
    recommendations: List[str]




@tool
def search_contract_clause(query: str) -> str:
    """Search for specific clauses or terms in the contract using semantic search.
    Use this to find relevant sections before analyzing risks."""
    # NOTE: The actual vector search is not done here.
    # We only use this tool so the LLM can REQUEST a search with a 'query' string.
    # The real search is executed in RAGContractAnalyzer._execute_tool.
    return query




class RAGContractAnalyzer:
    def __init__(self, vectorstore_path: str = "./chroma_db"):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(
            persist_directory=vectorstore_path,
            embedding_function=self.embeddings,
            collection_name="contracts",
        )

        # LLM with tool calling capability
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Register tools with the LLM 
        self.tools = [search_contract_clause]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute tool calls from LLM"""
        if tool_name == "search_contract_clause":
            query = tool_input.get("query", "")
            results = self.vectorstore.similarity_search(query, k=3)
            return "\n\n".join([doc.page_content for doc in results])
        return ""

    def analyze_with_rag(self, contract_text: str) -> dict:
        """Analyze contract using RAG approach"""

        print("Step 1: LLM identifying areas to investigate...")

        # First pass: LLM decides what to search for
        initial_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a contract analysis assistant. First, identify key areas 
that need investigation (payment terms, liability, termination, data usage, etc.).
Use the search_contract_clause tool to find relevant sections.""",
                ),
                (
                    "user",
                    "Analyze this contract. First, identify what areas to investigate:\n\n{contract_text}",
                ),
            ]
        )

        chain = initial_prompt | self.llm_with_tools
        # limit initial text length
        response = chain.invoke({"contract_text": contract_text[:2000]})

        # Execute any tool calls
        gathered_context = []
        tool_calls = getattr(response, "tool_calls", None)

        if tool_calls:
            print(f"Step 2: Executing {len(tool_calls)} searches...")
            for tool_call in tool_calls:
                name = tool_call["name"]
                args = tool_call["args"]
                result = self._execute_tool(name, args)
                gathered_context.append(f"Search: {args}\nResults: {result}")
                print(f"  - Searched: {args.get('query', 'N/A')}")

        # Second pass: Analyze with gathered context
        print("Step 3: Performing detailed analysis with retrieved context...")

        analysis_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert contract attorney. Analyze the contract thoroughly 
using both the full contract text and the retrieved relevant sections.

Identify:
1. Loopholes with specific clause references
2. Risks with severity scores (1-10)
3. Red flags
4. Negotiation recommendations

Be specific and reference exact clauses.""",
                ),
                (
                    "user",
                    """Full Contract:
{contract_text}

Retrieved Relevant Sections:
{context}

Provide a detailed analysis.""",
                ),
            ]
        )

        # Use structured output for final analysis
        final_llm = self.llm.with_structured_output(ContractAnalysis)
        final_chain = analysis_prompt | final_llm

        context_text = (
            "\n\n".join(gathered_context)
            if gathered_context
            else "No additional context retrieved"
        )

        analysis = final_chain.invoke(
            {
                "contract_text": contract_text,
                "context": context_text,
            }
        )

        return {
            "analysis": analysis,
            "retrieved_context": gathered_context,
            "tool_calls_made": len(tool_calls) if tool_calls else 0,
        }

    def print_rag_analysis(self, result: dict):
        """Print analysis with RAG information"""
        print(f"\n{'=' * 70}")
        print("RAG-ENHANCED CONTRACT ANALYSIS")
        print(f"{'=' * 70}")
        print(f"Tool Calls Made: {result['tool_calls_made']}")
        print(f"Context Pieces Retrieved: {len(result['retrieved_context'])}")

        analysis: ContractAnalysis = result["analysis"]

        print(f"\nOVERALL RISK SCORE: {analysis.overall_risk_score}/100")
        print(f"\nSUMMARY:\n{analysis.summary}")

        print(f"\n{'─' * 70}")
        print(f"LOOPHOLES ({len(analysis.loopholes)}):")
        for i, loophole in enumerate(analysis.loopholes, 1):
            print(f"\n{i}. [{loophole.severity}] {loophole.clause_reference}")
            print(f"   {loophole.loophole_description}")

        print(f"\n{'─' * 70}")
        print(f"TOP RISKS ({len(analysis.risks)}):")
        for i, risk in enumerate(analysis.risks, 1):
            print(f"\n{i}. {risk.risk_type} (Score: {risk.severity_score}/10)")
            print(f"   {risk.description}")

        print(f"\n{'─' * 70}")
        print(f"RED FLAGS: {len(analysis.red_flags)}")
        for flag in analysis.red_flags:
            print(f"  • {flag}")

        print(f"\n{'=' * 70}\n")




if __name__ == "__main__":
    print("=== Day 3: RAG Integration Test ===\n")

    # Load contract
    with open("test_contracts/sample_contract.txt", "r", encoding="utf-8") as f:
        contract_text = f.read()

    # Initialize RAG analyzer
    analyzer = RAGContractAnalyzer()

    print("Analyzing contract with RAG...\n")

    # Analyze
    result = analyzer.analyze_with_rag(contract_text)

    # Print results
    analyzer.print_rag_analysis(result)

    # Save results
    import json

    with open("test_contracts/rag_analysis.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tool_calls": result["tool_calls_made"],
                "analysis": result["analysis"].model_dump(),
            },
            f,
            indent=2,
        )
