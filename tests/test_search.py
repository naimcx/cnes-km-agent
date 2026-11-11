from src.rag_llamaindex import get_qdrant_retriever

def test_qdrant_retriever():
    """Test the Qdrant retriever with a set of queries"""

    retriever = get_qdrant_retriever(similarity_top_k=5, mode="hybrid", alpha=0.7)

    queries = [
        "Vinci engine cryogenic upper stage specific impulse",
        "Figure 5.3.1a Usable volume beneath payload fairing A62",
    ]

    for q in queries:
        print(f"\n================ QUERY: {q} ================")
        nodes = retriever.retrieve(q)
        assert len(nodes) > 0, f"No results found for query: {q}"
        print(f"Number of results : {len(nodes)}")
        for i, node in enumerate(nodes):
            page = node.metadata.get("page_label", "N/A")
            file_name = node.metadata.get("file_name", "N/A")
            score = node.score if hasattr(node, "score") and node.score is not None else "N/A"
            print(f"[{i+1}] Page {page} (File: {file_name}) - Score: {score}")
            print(f"Chunk : {node.text[:150]}...\n")
            img_path = node.metadata.get("image_path", "")
            has_drawings = node.metadata.get("has_drawings", False)
            print(f"Diagram detexted : {has_drawings} | Image : {img_path if img_path else 'None'}")
