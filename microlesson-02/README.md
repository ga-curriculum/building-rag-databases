<h1>
  <span class="headline">Building RAG Databases</span>
  <span class="subhead">Building the Complete RAG Pipeline</span>
</h1>

**Learning objective:** By the end of this lesson, students will be able to build a complete end-to-end RAG chain, implement effective retrieval and generation components, and optimize RAG parameters through experimentation.

---

## Step 3: Embed and Store Chunks in a Vector Store

Now comes the magic moment. We'll convert our text chunks into numerical vectors and store them in our in-memory database.

Add this code to a new cell:

```python
# =================================================================
#  Step 3: Embed and Store Chunks in a Vector Store
# =================================================================
# This single line is the magic. It replaces all the complex database connection code.
# It uses OpenAI's models to create embeddings and stores them in a FAISS vector store in memory.
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
print("In-memory vector store created successfully.")

# Let's peek behind the curtain - what did we just create?
print(f"Vector store contains {vectorstore.index.ntotal} vectors")
print(f"Each vector has {vectorstore.index.d} dimensions")
```

**What's happening here?**
- `OpenAIEmbeddings()`: Uses OpenAI's text-embedding-ada-002 model to convert text to vectors
- `FAISS.from_documents()`: This one line does the heavy lifting:
  - Sends each chunk to OpenAI's embedding API
  - Receives back a 1536-dimensional vector for each chunk
  - Stores all vectors in a FAISS index optimized for similarity search
- The entire process happens in memory—no external database required

**The Vector Magic:** Each support ticket is now represented as a point in 1536-dimensional space, positioned near similar tickets based on semantic meaning.

---

## Step 4: Create a Retriever and Test It

Before building the full chain, let's test our retrieval mechanism:

```python
# =================================================================
#  Step 4: Create a Retriever and Test It
# =================================================================
# The retriever's job is to fetch the relevant documents from the vector store.
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 most relevant chunks

query = "My payment failed, how do I update my credit card info?"
retrieved_docs = retriever.invoke(query)

print("--- Top 3 Retrieved Documents (The 'Retrieval' part of RAG) ---")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"\n{i}. Ticket ID: {doc.metadata['ticket_id']}, Status: {doc.metadata['status']}")
    print(f"   Content: {doc.page_content}")
    
# Let's also see the similarity scores
scores = vectorstore.similarity_search_with_score(query, k=3)
print("\n--- Similarity Scores (Lower = More Similar) ---")
for i, (doc, score) in enumerate(scores, 1):
    print(f"{i}. Ticket {doc.metadata['ticket_id']}: {score:.4f}")
```

**What's happening here?**
- `as_retriever()`: Converts our vector store into a retriever object
- `k=3`: Returns the 3 most similar documents to our query
- The retriever embeds our query using the same model and finds the closest vectors
- `similarity_search_with_score()`: Shows the actual distance scores (lower = more similar)

**Expected Result:** The retriever should find tickets T001, T005, and T011—all related to payment and credit card issues.

---

## Step 5: Augment and Generate (The Full RAG Chain)

Now we'll build the complete RAG chain that combines retrieval with generation:

```python
# =================================================================
#  Step 5: Augment and Generate (The Full RAG Chain)
# =================================================================
# Now we'll build the full RAG chain that takes the retrieved docs
# and uses an LLM to generate a human-friendly answer.

# Define a prompt template that instructs the LLM how to use the context
template = """
You are a helpful support assistant. Answer the user's question based ONLY on the following context.
If the context doesn't contain the answer, say that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# Define the LLM we'll use for generation
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Build the RAG chain using LangChain Expression Language (LCEL)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Invoke the chain with our query
print("\n--- Generating Final Answer (The 'Generation' part of RAG) ---")
final_answer = rag_chain.invoke(query)
print(final_answer)
```

**What's happening here?**
- **Prompt Template:** Instructs the LLM to use only the provided context
- **LangChain Expression Language (LCEL):** The `|` operator chains operations together
- **The Chain Flow:**
  1. `{"context": retriever, "question": RunnablePassthrough()}`: Retrieves relevant docs and passes the question
  2. `prompt`: Fills the template with context and question
  3. `llm`: Sends the prompt to GPT-3.5-Turbo
  4. `StrOutputParser()`: Extracts the text response

**Expected Result:** A natural language answer that references specific ticket information from the retrieved context.

---

## Deconstructing the Complete Code

Let's understand how all the pieces work together. Here's the complete pipeline in one view:

```python
# Complete RAG Pipeline - All Steps Together
print("=== COMPLETE RAG PIPELINE ===\n")

# Step 1: Query comes in
user_query = "My payment failed, how do I update my credit card info?"
print(f"User Query: {user_query}")

# Step 2: Query gets embedded (same model as documents)
query_embedding = embeddings.embed_query(user_query)
print(f"Query embedded into {len(query_embedding)} dimensions")

# Step 3: Vector search finds similar documents
similar_docs = vectorstore.similarity_search(user_query, k=3)
print(f"Found {len(similar_docs)} relevant documents")

# Step 4: Context is prepared for the LLM
context = "\n\n".join([doc.page_content for doc in similar_docs])
print(f"Context prepared ({len(context)} characters)")

# Step 5: LLM generates answer based on context
final_prompt = template.format(context=context, question=user_query)
response = llm.invoke([{"role": "user", "content": final_prompt}])
print(f"\nFinal Answer: {response.content}")
```

---

## Hands-On Challenge & Exploration

Now it's your turn to experiment. Modify the code in your Colab notebook to answer the following questions:

### Challenge 1: New Query Testing

Change the query to test different scenarios:

```python
# Test different queries
test_queries = [
    "How can I get a refund for a double charge?",
    "My app keeps crashing when syncing data",
    "I want to cancel my subscription",
    "How do I upgrade my storage plan?",
    "I'm not getting email notifications"
]

print("=== TESTING MULTIPLE QUERIES ===\n")
for i, test_query in enumerate(test_queries, 1):
    print(f"{i}. Query: {test_query}")
    answer = rag_chain.invoke(test_query)
    print(f"   Answer: {answer}\n")
```

**Discussion:** Does the retriever find the correct documents? Are the answers accurate and grounded in the context?

### Challenge 2: Tuning Retrieval Parameters

Experiment with different retrieval settings:

```python
# Test different retrieval parameters
def test_retrieval_parameters(query, k_values):
    for k in k_values:
        print(f"\n--- Retrieving Top {k} Documents ---")
        retriever_k = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever_k.invoke(query)
        
        print(f"Retrieved {len(docs)} documents:")
        for doc in docs:
            print(f"  - Ticket {doc.metadata['ticket_id']}: {doc.page_content[:50]}...")

test_query = "How can I get a refund for a double charge?"
test_retrieval_parameters(test_query, [1, 3, 5])
```

**Discussion:** How does changing `k` affect the final answer quality? Does more context always mean a better answer?

### Challenge 3: Prompt Engineering

Modify the prompt template to change the assistant's behavior:

```python
# Test different prompt styles
prompt_templates = {
    "brief": """Answer briefly and directly based on this context: {context}
Question: {question}
Brief Answer:""",
    
    "detailed": """You are a thorough support assistant. Provide a comprehensive answer based on the context below.
Include relevant ticket IDs and explain your reasoning.

Context: {context}
Question: {question}
Detailed Answer:""",
    
    "empathetic": """You are a caring and empathetic support assistant. Show understanding for the user's frustration.
Use the context to provide a helpful and compassionate response.

Context: {context}
Question: {question}
Empathetic Response:"""
}

query = "The mobile app crashes every time I try to sync my data"

for style, template in prompt_templates.items():
    print(f"\n--- {style.upper()} STYLE ---")
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answer = chain.invoke(query)
    print(answer)
```

**Discussion:** How does prompt engineering change the tone, length, and helpfulness of responses?

---

## Advanced: Understanding Vector Similarity

Let's peek behind the curtain to understand how vector similarity works:

```python
# Advanced: Exploring Vector Similarities
import numpy as np

def explore_vector_similarity():
    # Get embeddings for different queries
    queries = [
        "payment failed credit card",
        "app crashes mobile sync",
        "storage upgrade plan",
        "banana recipe cooking"  # Unrelated query
    ]
    
    print("=== VECTOR SIMILARITY EXPLORATION ===\n")
    
    for query in queries:
        print(f"Query: '{query}'")
        # Find similar documents
        results = vectorstore.similarity_search_with_score(query, k=2)
        
        for doc, score in results:
            print(f"  Ticket {doc.metadata['ticket_id']}: {score:.4f}")
            print(f"    Content: {doc.page_content[:60]}...")
        print()

explore_vector_similarity()
```

**Key Insight:** Notice how unrelated queries (like "banana recipe") get high similarity scores—they're mathematically distant from our support ticket embeddings.

---

## Production Considerations

Before moving to production, consider these real-world factors:

### 1. Embedding Costs and Caching

```python
# In production, cache embeddings to avoid re-computation costs
print("=== EMBEDDING COST CONSIDERATIONS ===")
print(f"Documents embedded: {len(chunks)}")
print(f"Approximate embedding cost: ${len(chunks) * 0.0001:.4f}")
print("💡 Tip: Cache embeddings and only re-embed when documents change")
```

### 2. Chunk Size Optimization

```python
# Test different chunk sizes
def test_chunk_sizes(sizes):
    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=50)
        test_chunks = splitter.split_documents(documents)
        print(f"Chunk size {size}: {len(test_chunks)} chunks created")

print("\n=== CHUNK SIZE OPTIMIZATION ===")
test_chunk_sizes([200, 500, 750, 1000])
```

### 3. Evaluation Metrics

```python
# Basic evaluation approach
def evaluate_rag_quality(test_cases):
    """
    In production, you'd want automated evaluation metrics:
    - Faithfulness: Does the answer stay true to the source?
    - Relevance: Does the answer address the question?
    - Context Precision: Are the retrieved documents relevant?
    """
    print("=== RAG QUALITY EVALUATION ===")
    for question, expected_tickets in test_cases.items():
        retrieved = retriever.invoke(question)
        retrieved_ids = [doc.metadata['ticket_id'] for doc in retrieved]
        
        print(f"\nQuestion: {question}")
        print(f"Expected tickets: {expected_tickets}")
        print(f"Retrieved tickets: {retrieved_ids}")
        
        # Simple precision calculation
        relevant_retrieved = len(set(expected_tickets) & set(retrieved_ids))
        precision = relevant_retrieved / len(retrieved_ids) if retrieved_ids else 0
        print(f"Precision@{len(retrieved_ids)}: {precision:.2f}")

# Test cases: question -> expected relevant ticket IDs
test_cases = {
    "My payment failed, how do I update my credit card?": ["T001", "T011"],
    "How can I get a refund for a double charge?": ["T005"],
    "The mobile app crashes when syncing": ["T009"]
}

evaluate_rag_quality(test_cases)
```

---

## Conclusion & Next Steps

Congratulations! You have successfully built a complete, end-to-end RAG pipeline using a streamlined, no-install approach. Let's review what you've accomplished:

### ✅ What You've Built

1. **Zero-Install Environment:** A complete RAG system running in your browser
2. **Data Pipeline:** CSV loading, chunking, and metadata preservation
3. **Vector Database:** In-memory FAISS store with semantic search capabilities
4. **Retrieval System:** Intelligent document retrieval based on semantic similarity
5. **Generation Chain:** LLM-powered answer generation grounded in retrieved context
6. **Evaluation Framework:** Basic metrics for assessing RAG quality

### 🔑 Key Insights

**RAG = Information Retrieval + Language Generation:** You've seen how combining vector search with LLMs creates a powerful system that can answer questions with perfect attribution to source documents.

**The Power of Simplicity:** Complex infrastructure isn't always necessary. Sometimes the best solution is the simplest one that works.

**Context is King:** The quality of your RAG system depends heavily on the quality of your retrieved context. Good chunking and retrieval strategies matter more than complex model architectures.

### 🚀 From Here to Production

To scale this to a production system, you'd need to consider:

- **Persistent Storage:** Replace FAISS with a production vector database (Pinecone, Weaviate, Chroma)
- **Batch Processing:** Embed large document collections efficiently
- **API Integration:** Wrap your RAG chain in a web service
- **Monitoring:** Track retrieval quality and answer relevance
- **Security:** Implement proper authentication and data protection

### 🎯 Looking Ahead

This knowledge base is a powerful tool. But a tool is only as good as the thinker using it. Our agent now has a memory, but it still needs to learn how to think more strategically.

In our next module, we will focus on **Agentic Thinking and Task Decomposition**. You'll learn how to design frameworks that allow an agent to break down a complex user request into a series of logical steps—steps that can be executed using the very RAG system you just built.

**The foundation is complete. Now let's teach our agent how to think.**

---

**🎉 Congratulations!** You've transformed from RAG theory to RAG practice. You now have the hands-on experience to build memory systems that can power intelligent agents in the real world.
