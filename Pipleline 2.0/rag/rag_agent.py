from typing import List, Dict, Any
from BaseAgent import BaseAgent
from .document_store import DocumentStore

class RAGAgent(BaseAgent):
    def __init__(self, role: str, basic_prompt: str, document_store: DocumentStore):
        super().__init__(role, basic_prompt)
        self.document_store = document_store
        self.retrieved_docs = []
    
    def search_knowledge_base(self, query: str, n_results: int = 5, filter_dict: Dict = None) -> List[Dict]:
        """Search the knowledge base for relevant information"""
        self.retrieved_docs = self.document_store.search_documents(
            query=query,
            n_results=n_results,
            filter_dict=filter_dict
        )
        return self.retrieved_docs
    
    def build_context_from_search(self, query: str, n_results: int = 5) -> str:
        """Build context string from search results"""
        search_results = self.search_knowledge_base(query, n_results)
        
        if not search_results:
            return "No relevant documents found in the knowledge base."
        
        context_parts = ["Retrieved Knowledge Base Information:"]
        
        for i, result in enumerate(search_results, 1):
            context_parts.append(f"\n--- Document {i} (Score: {result['score']:.3f}) ---")
            context_parts.append(f"Source: {result['metadata'].get('source', 'Unknown')}")
            context_parts.append(f"Content: {result['content']}")
        
        return "\n".join(context_parts)
    
    def get_output_with_rag(self, user_query: str = None, search_filter: Dict = None):
        """Get output enhanced with RAG"""
        if not self.llm:
            raise ValueError("LLM is not set.")
        
        # If no specific query provided, use the basic prompt as search query
        search_query = user_query or self.basic_prompt
        
        # Retrieve relevant documents
        rag_context = self.build_context_from_search(search_query, n_results=5)
        
        # Combine original context with RAG context
        enhanced_context = f"{self.context}\n\n{rag_context}" if self.context else rag_context
        
        # Use enhanced prompt if available
        base_prompt = self.enhanced_prompt if self.enhanced_prompt else self.basic_prompt
        
        final_prompt_template = (
            "You are an expert in {role}.\n\n"
            "CONTEXT AND KNOWLEDGE BASE:\n"
            "{enhanced_context}\n\n"
            "TASK:\n"
            "{base_prompt}\n\n"
            "Use the provided context and knowledge base information to give a comprehensive and accurate response. "
            "If the knowledge base contains relevant information, incorporate it into your response. "
            "If the knowledge base doesn't contain relevant information, rely on your general knowledge but mention this limitation."
        )
        
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        
        prompt = PromptTemplate(
            input_variables=["role", "enhanced_context", "base_prompt"],
            template=final_prompt_template
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.invoke({
            "role": self.role,
            "enhanced_context": enhanced_context,
            "base_prompt": base_prompt
        })['text']