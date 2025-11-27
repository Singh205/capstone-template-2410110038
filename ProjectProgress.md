Doc_processing :
---
This code sets up a complete workflow for loading, splitting, and embedding documents so they can be searched semantically. It loads PDF or text files, breaks them into manageable chunks, and stores their embeddings in a Chroma vector database. A sample contract is processed to test everything, and then semantic queries like payment terms or termination conditions are run against the vector store. Overall, it verifies that the document-processing pipeline works end-to-end.



Risk analysis:
---
This code builds contract analyzer that returns structured legal insights using Pydantic models. It defines clear schemas for loopholes, risks, red flags, and recommendations, and then uses an OpenAI model to fill these fields with detailed analysis. The contract text is processed through a custom prompt that guides the model to think like an expert attorney. Finally, the results are printed in a readable format and saved as a JSON file for later use.




Tool Calling:
---
This code mixes the contract analyzer with a RAG system so the AI can search the document before giving its final answer. First, the model decides what parts of the contract it needs to look up, then it uses semantic search to fetch those clauses. After that, it runs a full contract analysis using both the original text and the retrieved chunks. The result shows risk scores, loopholes, red flags, and more.
