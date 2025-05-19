from research_agent import ResearchAgent
import os

def test_research_agent():
    # Initialize agent
    agent = ResearchAgent()
    
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    
    # Copy the PDF to data directory
    pdf_path = "/run/media/magus/data/Masters/502/solow---2/Advanced Macroeconomics - Romer.pdf"
    os.system(f"cp '{pdf_path}' data/")
    
    # Ingest the PDF
    agent.ingest_documents("data")
    
    # Test queries
    queries = [
        "What are the key concepts of macroeconomics?",
        "What is the Solow growth model?",
        "What are the implications of the Solow residual?",
        "How does technological progress affect economic growth?"
        "What are the main topics covered in Advanced Macroeconomics?",
        "What are the key concepts discussed in the Solow growth model?",
        "What are some important economic theories mentioned in the book?"
    ]
    
    print("\nTesting research agent with queries:")
    for query in queries:
        print(f"\nQuery: {query}")
        result = agent.research(query)
        print(f"Response:\n{result}\n")

if __name__ == "__main__":
    test_research_agent()
