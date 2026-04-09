"""
query_expansion.py - Query Expansion for Failed Retrievals V9.0
==============================================================
✅ Expand queries when initial retrieval fails
✅ Multiple expansion strategies
✅ Semantic query reformulation
✅ Synonym and related term injection
"""

import re
from typing import List, Dict, Any, Tuple


class QueryExpansionEngine:
    """
    Query expansion to improve retrieval when initial query fails
    """
    
    def __init__(self):
        self.expansion_templates = {
            'question_reformulation': [
                "What is {query}?",
                "Explain {query}",
                "Define {query}",
                "{query} meaning",
                "Show information about {query}"
            ],
            'context_addition': [
                "{query} in the document",
                "{query} according to the paper",
                "{query} details",
                "{query} explanation"
            ],
            'specificity_reduction': [
                "general information about {main_concept}",
                "{main_concept} overview",
                "background on {main_concept}"
            ],
            'component_splitting': [
                "{part1}",
                "{part2}",
                "{part1} and {part2}"
            ]
        }
        
        self.domain_synonyms = {
            'equation': ['formula', 'expression', 'mathematical relation'],
            'table': ['data', 'results', 'values', 'numbers'],
            'figure': ['image', 'diagram', 'illustration', 'graph', 'plot'],
            'model': ['architecture', 'framework', 'system', 'approach'],
            'method': ['technique', 'approach', 'strategy', 'algorithm'],
            'result': ['outcome', 'finding', 'conclusion', 'performance'],
            'compare': ['difference', 'contrast', 'versus', 'comparison'],
            'rag': ['retrieval augmented generation', 'retrieval-based generation'],
            'transformer': ['attention mechanism', 'self-attention model'],
            'embedding': ['vector representation', 'dense vector'],
        }
    
    def expand_query(
        self, 
        original_query: str, 
        expansion_strategy: str = 'multi_strategy',
        max_expansions: int = 5
    ) -> List[str]:
        """
        Expand query using specified strategy
        
        Args:
            original_query: Original failed query
            expansion_strategy: Strategy to use
                - 'question_reformulation': Rephrase as different questions
                - 'synonym_injection': Add synonyms
                - 'context_addition': Add context phrases
                - 'specificity_reduction': Make query more general
                - 'multi_strategy': Combine multiple strategies
            max_expansions: Maximum number of expanded queries
        
        Returns:
            List of expanded queries
        """
        if expansion_strategy == 'question_reformulation':
            return self._reformulate_as_questions(original_query, max_expansions)
        elif expansion_strategy == 'synonym_injection':
            return self._inject_synonyms(original_query, max_expansions)
        elif expansion_strategy == 'context_addition':
            return self._add_context(original_query, max_expansions)
        elif expansion_strategy == 'specificity_reduction':
            return self._reduce_specificity(original_query, max_expansions)
        elif expansion_strategy == 'multi_strategy':
            return self._multi_strategy_expansion(original_query, max_expansions)
        else:
            return [original_query]
    
    def _reformulate_as_questions(self, query: str, max_count: int) -> List[str]:
        """Reformulate as different question types"""
        expansions = []
        
        query_clean = query.strip().rstrip('?')
        
        if not self._is_question(query):
            for template in self.expansion_templates['question_reformulation'][:max_count]:
                expanded = template.format(query=query_clean)
                expansions.append(expanded)
        else:
            question_types = [
                ('what', 'how'),
                ('how', 'what'),
                ('why', 'what'),
                ('explain', 'describe')
            ]
            
            query_lower = query.lower()
            for old, new in question_types:
                if old in query_lower:
                    new_query = re.sub(
                        rf'\b{old}\b', 
                        new, 
                        query_lower, 
                        count=1, 
                        flags=re.IGNORECASE
                    )
                    expansions.append(new_query)
                    if len(expansions) >= max_count:
                        break
        
        return expansions[:max_count]
    
    def _inject_synonyms(self, query: str, max_count: int) -> List[str]:
        """Inject domain-specific synonyms"""
        expansions = []
        query_lower = query.lower()
        
        for term, synonyms in self.domain_synonyms.items():
            if term in query_lower:
                for syn in synonyms:
                    new_query = re.sub(
                        rf'\b{term}\b', 
                        syn, 
                        query_lower, 
                        count=1,
                        flags=re.IGNORECASE
                    )
                    if new_query != query_lower:
                        expansions.append(new_query)
                        if len(expansions) >= max_count:
                            return expansions
        
        if not expansions:
            expansions = [query]
        
        return expansions[:max_count]
    
    def _add_context(self, query: str, max_count: int) -> List[str]:
        """Add contextual phrases"""
        expansions = []
        
        for template in self.expansion_templates['context_addition'][:max_count]:
            expanded = template.format(query=query.strip())
            expansions.append(expanded)
        
        return expansions
    
    def _reduce_specificity(self, query: str, max_count: int) -> List[str]:
        """Make query more general by extracting main concepts"""
        expansions = []
        
        main_concepts = self._extract_main_concepts(query)
        
        if main_concepts:
            for concept in main_concepts[:max_count]:
                for template in self.expansion_templates['specificity_reduction']:
                    expanded = template.format(main_concept=concept)
                    expansions.append(expanded)
                    if len(expansions) >= max_count:
                        return expansions
        
        if not expansions:
            words = query.split()
            if len(words) > 3:
                shorter_query = ' '.join(words[:len(words)//2])
                expansions.append(shorter_query)
        
        return expansions[:max_count]
    
    def _multi_strategy_expansion(self, query: str, max_count: int) -> List[str]:
        """Combine multiple expansion strategies"""
        all_expansions = []
        
        strategies = [
            ('reformulate_as_questions', 2),
            ('inject_synonyms', 2),
            ('add_context', 1)
        ]
        
        for strategy, count in strategies:
            method = getattr(self, f'_{strategy}')
            expansions = method(query, count)
            all_expansions.extend(expansions)
            
            if len(all_expansions) >= max_count:
                break
        
        unique_expansions = []
        seen = set()
        
        for exp in all_expansions:
            exp_lower = exp.lower().strip()
            if exp_lower not in seen:
                seen.add(exp_lower)
                unique_expansions.append(exp)
        
        return unique_expansions[:max_count]
    
    def _is_question(self, query: str) -> bool:
        """Check if query is a question"""
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which']
        query_lower = query.lower()
        
        return (
            any(query_lower.startswith(qw) for qw in question_words) or
            '?' in query
        )
    
    def _extract_main_concepts(self, query: str) -> List[str]:
        """Extract main concepts from query"""
        stopwords = {
            'what', 'how', 'why', 'when', 'where', 'who', 'which',
            'is', 'are', 'the', 'a', 'an', 'of', 'in', 'to', 'for',
            'show', 'explain', 'tell', 'me', 'about'
        }
        
        words = re.findall(r'\b[a-z]+\b', query.lower())
        
        concepts = [w for w in words if w not in stopwords and len(w) > 3]
        
        bigrams = []
        for i in range(len(concepts) - 1):
            bigram = f"{concepts[i]} {concepts[i+1]}"
            bigrams.append(bigram)
        
        all_concepts = bigrams + concepts
        
        return all_concepts[:5]
    
    def expand_with_failed_terms(
        self, 
        query: str, 
        failed_retrieval_scores: List[float],
        min_score_threshold: float = 0.5
    ) -> List[str]:
        """
        Expand query based on failed retrieval scores
        
        If retrieval scores are low, use more aggressive expansion
        """
        avg_score = sum(failed_retrieval_scores) / len(failed_retrieval_scores) if failed_retrieval_scores else 0
        
        if avg_score < min_score_threshold * 0.5:
            strategy = 'specificity_reduction'
            max_count = 5
        elif avg_score < min_score_threshold:
            strategy = 'multi_strategy'
            max_count = 4
        else:
            strategy = 'synonym_injection'
            max_count = 3
        
        return self.expand_query(query, strategy, max_count)


class AdaptiveQueryExpansion:
    """
    Adaptive query expansion that learns from retrieval success
    """
    
    def __init__(self):
        self.expansion_engine = QueryExpansionEngine()
        self.success_history: Dict[str, List[Tuple[str, float]]] = {}
    
    def expand_adaptively(
        self, 
        original_query: str,
        previous_score: float = 0.0,
        retrieval_attempt: int = 1
    ) -> List[str]:
        """
        Adaptively expand query based on previous attempts
        
        Args:
            original_query: Original query
            previous_score: Score from previous retrieval attempt
            retrieval_attempt: Number of retrieval attempts so far
        
        Returns:
            List of expanded queries, ordered by likely success
        """
        if retrieval_attempt == 1:
            if previous_score < 0.3:
                strategy = 'multi_strategy'
                max_count = 5
            elif previous_score < 0.6:
                strategy = 'synonym_injection'
                max_count = 3
            else:
                strategy = 'question_reformulation'
                max_count = 2
        
        elif retrieval_attempt == 2:
            strategy = 'specificity_reduction'
            max_count = 4
        
        else:
            similar_queries = self._find_similar_successful_queries(original_query)
            if similar_queries:
                return [q for q, _ in similar_queries[:3]]
            else:
                strategy = 'context_addition'
                max_count = 3
        
        expansions = self.expansion_engine.expand_query(
            original_query, 
            strategy, 
            max_count
        )
        
        return expansions
    
    def record_success(self, query: str, expansion: str, score: float):
        """Record successful expansion for future learning"""
        if query not in self.success_history:
            self.success_history[query] = []
        
        self.success_history[query].append((expansion, score))
        
        self.success_history[query].sort(key=lambda x: -x[1])
        
        if len(self.success_history[query]) > 10:
            self.success_history[query] = self.success_history[query][:10]
    
    def _find_similar_successful_queries(self, query: str) -> List[Tuple[str, float]]:
        """Find similar queries that succeeded in the past"""
        query_words = set(query.lower().split())
        
        similar = []
        for past_query, expansions in self.success_history.items():
            past_words = set(past_query.lower().split())
            
            overlap = len(query_words & past_words)
            similarity = overlap / max(len(query_words), len(past_words))
            
            if similarity > 0.5 and expansions:
                best_expansion, best_score = expansions[0]
                similar.append((best_expansion, best_score * similarity))
        
        similar.sort(key=lambda x: -x[1])
        
        return similar


def test_query_expansion():
    """Test query expansion engine"""
    engine = QueryExpansionEngine()
    
    print("=" * 70)
    print("Query Expansion Engine - Test Results")
    print("=" * 70)
    
    test_queries = [
        ("What is RAG?", "question_reformulation"),
        ("Show equation 5", "synonym_injection"),
        ("Compare table 1 and equation 3", "multi_strategy"),
        ("Transformer architecture details", "context_addition"),
        ("self-attention mechanism in transformers", "specificity_reduction"),
    ]
    
    for query, strategy in test_queries:
        print(f"\n{'─' * 70}")
        print(f"📝 Original Query: {query}")
        print(f"🔧 Strategy: {strategy}")
        print(f"{'─' * 70}")
        
        expansions = engine.expand_query(query, strategy, max_expansions=5)
        
        print(f"✨ Expanded Queries ({len(expansions)}):")
        for i, exp in enumerate(expansions, 1):
            print(f"   {i}. {exp}")
    
    print(f"\n{'=' * 70}")
    print("Adaptive Expansion Test")
    print(f"{'=' * 70}")
    
    adaptive = AdaptiveQueryExpansion()
    
    query = "What is self-RAG?"
    print(f"\n📝 Query: {query}")
    
    for attempt in range(1, 4):
        score = 0.2 * attempt
        print(f"\n🔄 Attempt {attempt} (previous score: {score:.2f})")
        
        expansions = adaptive.expand_adaptively(query, score, attempt)
        
        print(f"   Expansions:")
        for i, exp in enumerate(expansions, 1):
            print(f"      {i}. {exp}")
        
        if attempt < 3:
            adaptive.record_success(query, expansions[0], score + 0.3)
    
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    test_query_expansion()