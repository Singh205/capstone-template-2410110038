# Day 2: Structured Output & Risk Analysis
# pip install langchain langchain-openai pydantic

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()

# Define structured output models
class Loophole(BaseModel):
    """A potential loophole in the contract"""
    clause_reference: str = Field(description="The specific clause or section")
    loophole_description: str = Field(description="Description of the loophole")
    severity: str = Field(description="HIGH, MEDIUM, or LOW")
    exploitation_scenario: str = Field(description="How this could be exploited")

class RiskItem(BaseModel):
    """A specific risk identified in the contract"""
    risk_type: str = Field(description="Type of risk (financial, legal, operational, etc.)")
    description: str = Field(description="Detailed description of the risk")
    severity_score: int = Field(description="Risk score from 1-10", ge=1, le=10)
    affected_party: str = Field(description="Who is affected (client, provider, both)")
    mitigation: Optional[str] = Field(description="Suggested mitigation strategy")

class ContractAnalysis(BaseModel):
    """Complete contract analysis with risk assessment"""
    overall_risk_score: int = Field(description="Overall risk score 1-100", ge=1, le=100)
    summary: str = Field(description="Brief summary of the contract")
    loopholes: List[Loophole] = Field(description="List of identified loopholes")
    risks: List[RiskItem] = Field(description="List of identified risks")
    red_flags: List[str] = Field(description="Major red flags or concerns")
    recommendations: List[str] = Field(description="Recommendations for negotiation")


class ContractAnalyzer:
    def __init__(self):
        # Use structured output with Pydantic
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        ).with_structured_output(ContractAnalysis)
        
        self.analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert contract attorney specializing in identifying risks, 
            loopholes, and unfair terms in contracts. Analyze the contract thoroughly and identify:
            
            1. Loopholes that could be exploited by either party
            2. Risks categorized by type (financial, legal, operational, reputational)
            3. Red flags or highly unfavorable terms
            4. Recommendations for negotiation
            
            Be thorough and specific. Reference exact clauses when identifying issues.
            Rate severity objectively based on potential impact."""),
            ("user", "Analyze this contract:\n\n{contract_text}")
        ])
    
    def analyze_contract(self, contract_text: str) -> ContractAnalysis:
        """Analyze contract and return structured results"""
        chain = self.analysis_prompt | self.llm
        result = chain.invoke({"contract_text": contract_text})
        return result
    
    def print_analysis(self, analysis: ContractAnalysis):
        """Pretty print the analysis"""
        print(f"\n{'='*70}")
        print(f"OVERALL RISK SCORE: {analysis.overall_risk_score}/100")
        print(f"{'='*70}")
        
        print(f"\nSUMMARY:\n{analysis.summary}")
        
        print(f"\n{'─'*70}")
        print(f"LOOPHOLES IDENTIFIED ({len(analysis.loopholes)}):")
        print(f"{'─'*70}")
        for i, loophole in enumerate(analysis.loopholes, 1):
            print(f"\n{i}. [{loophole.severity}] {loophole.clause_reference}")
            print(f"   Description: {loophole.loophole_description}")
            print(f"   Exploitation: {loophole.exploitation_scenario}")
        
        print(f"\n{'─'*70}")
        print(f"RISK ASSESSMENT ({len(analysis.risks)}):")
        print(f"{'─'*70}")
        for i, risk in enumerate(analysis.risks, 1):
            print(f"\n{i}. {risk.risk_type.upper()} - Score: {risk.severity_score}/10")
            print(f"   Affects: {risk.affected_party}")
            print(f"   Description: {risk.description}")
            if risk.mitigation:
                print(f"   Mitigation: {risk.mitigation}")
        
        print(f"\n{'─'*70}")
        print(f"RED FLAGS ({len(analysis.red_flags)}):")
        print(f"{'─'*70}")
        for i, flag in enumerate(analysis.red_flags, 1):
            print(f"{i}. {flag}")
        
        print(f"\n{'─'*70}")
        print(f"RECOMMENDATIONS ({len(analysis.recommendations)}):")
        print(f"{'─'*70}")
        for i, rec in enumerate(analysis.recommendations, 1):
            print(f"{i}. {rec}")
        
        print(f"\n{'='*70}\n")


# Test the analyzer
if __name__ == "__main__":
    print("=== Day 2: Structured Analysis Test ===\n")
    
    # Read the sample contract from Day 1
    with open("test_contracts/sample_contract.txt", "r") as f:
        contract_text = f.read()
    
    # Initialize analyzer
    analyzer = ContractAnalyzer()
    
    print("Analyzing contract... (this may take 10-20 seconds)")
    
    # Analyze contract
    analysis = analyzer.analyze_contract(contract_text)
    
    # Print results
    analyzer.print_analysis(analysis)
    
    # Save analysis as JSON
    import json
    with open("test_contracts/analysis_result.json", "w") as f:
        json.dump(analysis.model_dump(), f, indent=2)
    
    print("✓ Day 2 Complete! Analysis saved to analysis_result.json")