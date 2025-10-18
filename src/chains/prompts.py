"""
Prompt templates for RAG summarization and question answering.
"""

from langchain_core.prompts import ChatPromptTemplate


def get_summarization_prompt() -> ChatPromptTemplate:
    """
    Get the prompt template for summarization tasks.
    
    Returns:
        ChatPromptTemplate for summarization
    """
    template = """You are a helpful AI assistant that creates accurate and concise summaries based on provided context.

Your task is to summarize information related to the user's query using ONLY the context provided below. 

Guidelines:
- Use only the information from the provided context
- If the context doesn't contain relevant information for the query, say so clearly
- Create a well-structured summary with clear points
- Be concise but comprehensive
- Maintain accuracy and avoid speculation
- If multiple sources are provided, synthesize the information coherently

Context:
{context}

Query: {query}

Summary:"""

    return ChatPromptTemplate.from_template(template)


def get_qa_prompt() -> ChatPromptTemplate:
    """
    Get the prompt template for question answering tasks.
    
    Returns:
        ChatPromptTemplate for Q&A
    """
    template = """You are a helpful AI assistant that answers questions based on provided context.

Your task is to answer the user's question using ONLY the information from the provided context below.

Guidelines:
- Use only the information from the provided context
- If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question"
- Be direct and concise in your answer
- If the answer requires multiple parts, structure it clearly
- Maintain accuracy and avoid speculation
- If multiple sources provide conflicting information, mention this

Context:
{context}

Question: {question}

Answer:"""

    return ChatPromptTemplate.from_template(template)


def get_detailed_summarization_prompt() -> ChatPromptTemplate:
    """
    Get a more detailed prompt template for comprehensive summarization.
    
    Returns:
        ChatPromptTemplate for detailed summarization
    """
    template = """You are an expert AI assistant specializing in creating comprehensive and well-structured summaries.

Your task is to create a detailed summary based on the user's query using ONLY the context provided below.

Guidelines:
- Use only the information from the provided context
- If the context doesn't contain relevant information for the query, say so clearly
- Create a structured summary with:
  * Key points (bullet points or numbered list)
  * Important details and examples
  * Any relevant statistics or data points
  * Conclusions or implications if applicable
- Be comprehensive but well-organized
- Maintain accuracy and avoid speculation
- Synthesize information from multiple sources when available

Context:
{context}

Query: {query}

Detailed Summary:"""

    return ChatPromptTemplate.from_template(template)


def get_bullet_point_prompt() -> ChatPromptTemplate:
    """
    Get a prompt template for bullet-point style summaries.
    
    Returns:
        ChatPromptTemplate for bullet-point summaries
    """
    template = """You are a helpful AI assistant that creates clear, bullet-point summaries.

Your task is to summarize information related to the user's query using ONLY the context provided below, formatted as bullet points.

Guidelines:
- Use only the information from the provided context
- If the context doesn't contain relevant information for the query, say so clearly
- Format your response as clear bullet points
- Each bullet point should be concise but informative
- Use sub-bullets when appropriate for organization
- Maintain accuracy and avoid speculation
- Prioritize the most important information

Context:
{context}

Query: {query}

Summary (bullet points):"""

    return ChatPromptTemplate.from_template(template)


def get_faq_prompt() -> ChatPromptTemplate:
    """
    Get a prompt template specifically for FAQ-style responses.
    
    Returns:
        ChatPromptTemplate for FAQ responses
    """
    template = """You are a helpful AI assistant that provides clear, FAQ-style answers.

Your task is to answer the user's question using ONLY the information from the provided context below, in a clear and helpful FAQ format.

Guidelines:
- Use only the information from the provided context
- If the context doesn't contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question"
- Provide a clear, direct answer
- If the answer involves steps or processes, number them
- Be helpful and user-friendly in tone
- Maintain accuracy and avoid speculation

Context:
{context}

Question: {question}

Answer:"""

    return ChatPromptTemplate.from_template(template)
