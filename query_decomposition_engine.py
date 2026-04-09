"""
query_decomposition_engine.py - Advanced Query Analysis V9.0
============================================================
✅ Multi-step reasoning for complex questions
✅ Comparative query handling
✅ Sequential query execution
✅ Answer fusion from multiple sub-queries
"""

import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class SubQuery:
    """Represents a decomposed sub-query"""
    query: str
    query_type: str
    priority: int
    dependencies: List[int] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class QueryDecompositionEngine:
    """
    Advanced query decomposition for complex research questions
    """
    
    def __init__(self):
        self.comparative_patterns = [
            (r'compare\s+(.+?)\s+(?:and|with|to)\s+(.+?)(?:\s|$)', 'comparison'),
            (r'difference\s+(?:between\s+)?(.+?)\s+and\s+(.+?)(?:\s|$)', 'difference'),
            (r'relation(?:ship)?\s+between\s+(.+?)\s+and\s+(.+?)(?:\s|$)', 'relation'),
            (r'(.+?)\s+versus\s+(.+?)(?:\s|$)', 'versus'),
            (r'(.+?)\s+vs\.?\s+(.+?)(?:\s|$)', 'versus'),
            (r'how\s+(?:does|do)\s+(.+?)\s+(?:differ|compare)\s+(?:from|to|with)\s+(.+?)(?:\s|$)', 'comparison'),
        ]
        
        self.sequential_patterns = [
            (r'(.+?),\s*then\s+(.+)', 'sequential'),
            (r'first\s+(.+?),\s*(?:then|and then|next)\s+(.+)', 'sequential'),
            (r'(.+?)\s+and\s+also\s+(.+)', 'additive'),
            (r'(.+?)\s+as\s+well\s+as\s+(.+)', 'additive'),
        ]
        
        self.reference_patterns = [
            (r'(?:table|tbl)\s+(\d+)\s+and\s+(?:equation|eq)\s+(\d+)', 'table_equation'),
            (r'(?:figure|fig)\s+(\d+)\s+and\s+(?:table|tbl)\s+(\d+)', 'figure_table'),
            (r'(?:equation|eq)\s+(\d+)\s+and\s+(?:equation|eq)\s+(\d+)', 'equation_equation'),
            (r'results?\s+(?:of|from)\s+(?:table|tbl)\s+(\d+)', 'table_results'),
            (r'(?:using|based on)\s+(?:equation|eq)\s+(\d+)', 'equation_based'),
        ]
    
    def decompose(self, query: str) -> Tuple[bool, List[SubQuery], str]:
        """
        Decompose complex query into sub-queries
        
        Returns:
            (is_complex, sub_queries, query_type)
        """
        query_lower = query.lower().strip()
        
        for pattern, qtype in self.comparative_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                return self._handle_comparative(query, match, qtype)
        
        for pattern, qtype in self.reference_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                return self._handle_reference(query, match, qtype)
        
        for pattern, qtype in self.sequential_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                return self._handle_sequential(query, match, qtype)
        
        return False, [SubQuery(query, 'simple', 1)], 'simple'
    
    def _handle_comparative(self, query: str, match: re.Match, qtype: str) -> Tuple[bool, List[SubQuery], str]:
        """Handle comparative queries like 'compare X and Y'"""
        entity1 = match.group(1).strip()
        entity2 = match.group(2).strip()
        
        sub_queries = [
            SubQuery(
                query=f"What is {entity1}?",
                query_type='retrieval',
                priority=1,
                dependencies=[]
            ),
            SubQuery(
                query=f"What is {entity2}?",
                query_type='retrieval',
                priority=1,
                dependencies=[]
            ),
            SubQuery(
                query=f"Compare {entity1} and {entity2}",
                query_type='synthesis',
                priority=2,
                dependencies=[0, 1]
            )
        ]
        
        return True, sub_queries, qtype
    
    def _handle_reference(self, query: str, match: re.Match, qtype: str) -> Tuple[bool, List[SubQuery], str]:
        """Handle queries referencing specific elements like 'table 1 and equation 3'"""
        if qtype == 'table_equation':
            table_num = match.group(1)
            eq_num = match.group(2)
            
            sub_queries = [
                SubQuery(
                    query=f"Show table {table_num}",
                    query_type='table_retrieval',
                    priority=1
                ),
                SubQuery(
                    query=f"Show equation {eq_num}",
                    query_type='equation_retrieval',
                    priority=1
                ),
                SubQuery(
                    query=f"Relate the results from table {table_num} to equation {eq_num}",
                    query_type='synthesis',
                    priority=2,
                    dependencies=[0, 1]
                )
            ]
            
        elif qtype == 'figure_table':
            fig_num = match.group(1)
            tbl_num = match.group(2)
            
            sub_queries = [
                SubQuery(
                    query=f"Show figure {fig_num}",
                    query_type='figure_retrieval',
                    priority=1
                ),
                SubQuery(
                    query=f"Show table {tbl_num}",
                    query_type='table_retrieval',
                    priority=1
                ),
                SubQuery(
                    query=f"How do figure {fig_num} and table {tbl_num} relate?",
                    query_type='synthesis',
                    priority=2,
                    dependencies=[0, 1]
                )
            ]
            
        elif qtype == 'equation_equation':
            eq1 = match.group(1)
            eq2 = match.group(2)
            
            sub_queries = [
                SubQuery(
                    query=f"Show equation {eq1}",
                    query_type='equation_retrieval',
                    priority=1
                ),
                SubQuery(
                    query=f"Show equation {eq2}",
                    query_type='equation_retrieval',
                    priority=1
                ),
                SubQuery(
                    query=f"What is the relationship between equation {eq1} and equation {eq2}?",
                    query_type='synthesis',
                    priority=2,
                    dependencies=[0, 1]
                )
            ]
            
        elif qtype == 'table_results':
            tbl_num = match.group(1)
            sub_queries = [
                SubQuery(
                    query=f"Show table {tbl_num} with all details",
                    query_type='table_retrieval',
                    priority=1
                )
            ]
            
        elif qtype == 'equation_based':
            eq_num = match.group(1)
            sub_queries = [
                SubQuery(
                    query=f"Show equation {eq_num}",
                    query_type='equation_retrieval',
                    priority=1
                ),
                SubQuery(
                    query=query,
                    query_type='synthesis',
                    priority=2,
                    dependencies=[0]
                )
            ]
        else:
            return False, [SubQuery(query, 'simple', 1)], 'simple'
        
        return True, sub_queries, qtype
    
    def _handle_sequential(self, query: str, match: re.Match, qtype: str) -> Tuple[bool, List[SubQuery], str]:
        """Handle sequential queries like 'first X, then Y'"""
        part1 = match.group(1).strip()
        part2 = match.group(2).strip()
        
        sub_queries = [
            SubQuery(
                query=part1,
                query_type='retrieval',
                priority=1,
                dependencies=[]
            ),
            SubQuery(
                query=part2,
                query_type='retrieval',
                priority=2,
                dependencies=[0]
            )
        ]
        
        return True, sub_queries, qtype
    
    def get_execution_plan(self, sub_queries: List[SubQuery]) -> List[List[int]]:
        """
        Get execution plan for sub-queries based on dependencies
        
        Returns list of execution batches (queries in same batch can run in parallel)
        """
        executed = set()
        plan = []
        
        max_priority = max(sq.priority for sq in sub_queries)
        
        for priority in range(1, max_priority + 1):
            batch = []
            for i, sq in enumerate(sub_queries):
                if sq.priority == priority and i not in executed:
                    if all(dep in executed for dep in sq.dependencies):
                        batch.append(i)
                        executed.add(i)
            
            if batch:
                plan.append(batch)
        
        return plan
    
    def should_use_fusion(self, query_type: str) -> bool:
        """Determine if answer fusion should be used"""
        return query_type in ['comparison', 'difference', 'relation', 'synthesis']


class AnswerFusion:
    """
    Fuse multiple answers from sub-queries into coherent final answer
    """
    
    @staticmethod
    def fuse_answers(
        sub_answers: List[Tuple[str, SubQuery]], 
        original_query: str,
        fusion_method: str = 'concatenate'
    ) -> str:
        """
        Fuse multiple sub-answers into final answer
        
        Methods:
            - concatenate: Simple concatenation with headers
            - synthesis: Generate synthesis prompt for LLM
            - weighted: Weight answers by relevance
        """
        if fusion_method == 'concatenate':
            return AnswerFusion._concatenate_fusion(sub_answers, original_query)
        elif fusion_method == 'synthesis':
            return AnswerFusion._synthesis_fusion(sub_answers, original_query)
        elif fusion_method == 'weighted':
            return AnswerFusion._weighted_fusion(sub_answers, original_query)
        else:
            return AnswerFusion._concatenate_fusion(sub_answers, original_query)
    
    @staticmethod
    def _concatenate_fusion(sub_answers: List[Tuple[str, SubQuery]], original_query: str) -> str:
        """Simple concatenation with section headers"""
        parts = []
        
        for i, (answer, sub_query) in enumerate(sub_answers, 1):
            if sub_query.query_type == 'synthesis':
                parts.append(answer)
            else:
                parts.append(f"**Part {i} ({sub_query.query_type}):**\n{answer}")
        
        return "\n\n".join(parts)
    
    @staticmethod
    def _synthesis_fusion(sub_answers: List[Tuple[str, SubQuery]], original_query: str) -> str:
        """Generate synthesis prompt for LLM"""
        context = "\n\n".join([
            f"**Context {i+1}:** {answer}" 
            for i, (answer, _) in enumerate(sub_answers)
        ])
        
        synthesis_prompt = f"""
Based on the following context pieces, provide a comprehensive answer to the question: "{original_query}"

{context}

**Synthesized Answer:**
"""
        return synthesis_prompt
    
    @staticmethod
    def _weighted_fusion(sub_answers: List[Tuple[str, SubQuery]], original_query: str) -> str:
        """Weight answers by priority and type"""
        synthesis_answers = []
        retrieval_answers = []
        
        for answer, sub_query in sub_answers:
            if sub_query.query_type == 'synthesis':
                synthesis_answers.append(answer)
            else:
                retrieval_answers.append(answer)
        
        final_parts = []
        
        if retrieval_answers:
            final_parts.append("**Retrieved Information:**\n" + "\n\n".join(retrieval_answers))
        
        if synthesis_answers:
            final_parts.append("**Analysis:**\n" + "\n\n".join(synthesis_answers))
        
        return "\n\n".join(final_parts)


def test_decomposition():
    """Test query decomposition engine"""
    engine = QueryDecompositionEngine()
    
    test_queries = [
        "Compare table 1 and equation 3",
        "What is the difference between RAG and traditional retrieval?",
        "Show equation 5 and then explain its application",
        "Relation between figure 2 and table 4",
        "Simple query about transformers"
    ]
    
    print("=" * 60)
    print("Query Decomposition Engine - Test Results")
    print("=" * 60)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        is_complex, sub_queries, qtype = engine.decompose(query)
        
        print(f"   Complex: {is_complex}")
        print(f"   Type: {qtype}")
        print(f"   Sub-queries ({len(sub_queries)}):")
        
        for i, sq in enumerate(sub_queries):
            print(f"      {i+1}. [{sq.query_type}] {sq.query}")
            if sq.dependencies:
                print(f"         Dependencies: {sq.dependencies}")
        
        if is_complex:
            plan = engine.get_execution_plan(sub_queries)
            print(f"   Execution Plan: {plan}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_decomposition()