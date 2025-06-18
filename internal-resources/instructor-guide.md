<h1>
  <span class="headline">Building RAG Databases</span>
  <span class="subhead">Instructor Guide</span>
</h1>

## Session Overview

This is a 90-minute hands-on lab designed to take students from RAG theory to practice. The session uses a zero-install approach with Google Colab to eliminate setup complexity and focus entirely on RAG implementation logic.

### Learning Objectives

By the end of this session, students will:
- Build a complete RAG pipeline from scratch using browser-based tools
- Understand the five core RAG steps: Load → Chunk → Embed → Retrieve → Generate
- Implement real-world RAG optimization techniques
- Evaluate RAG system performance using practical metrics

## Pre-Session Preparation

### Required Setup (Students)
1. **OpenAI API Key:** Students need active OpenAI API keys
2. **Google Account:** For Colab access (free tier sufficient)
3. **Browser:** Modern browser with JavaScript enabled

### Instructor Preparation
1. Test the complete code pipeline in Colab beforehand
2. Prepare backup API keys for students who encounter issues
3. Review common Colab troubleshooting scenarios
4. Have the support ticket dataset ready to examine

## Session Structure & Timing

### Microlesson 1: Setting Up Your RAG Environment (45 minutes)
- **0-10 min:** Lab objectives and no-install approach explanation
- **10-20 min:** Colab setup and API key configuration  
- **20-35 min:** Code through Steps 0-2 (install, load, chunk)
- **35-45 min:** RAG architecture overview and Q&A

### Microlesson 2: Building the Complete RAG Pipeline (45 minutes)
- **0-15 min:** Steps 3-5 (embed, retrieve, generate)
- **15-30 min:** Hands-on challenges and experimentation
- **30-40 min:** Production considerations and evaluation
- **40-45 min:** Wrap-up and next steps preview

## Key Teaching Points

### The "No-Install" Philosophy
Emphasize that this approach eliminates infrastructure distractions. Students focus on RAG logic, not DevOps. This is intentional pedagogical design.

### Code-First Learning
Students see the complete working system first, then deconstruct it. This builds confidence before diving into details.

### Semantic Search Magic
Help students understand that vector similarity is mathematical—similar concepts cluster together in high-dimensional space.

### Context is King
Reinforce that RAG quality depends more on good retrieval than complex generation models.

## Common Student Challenges

### API Key Issues
- **Problem:** Students forget to enable "Notebook access" toggle
- **Solution:** Show the exact Colab secrets interface steps

### Embedding Costs Confusion
- **Problem:** Students worry about API costs
- **Solution:** Explain the dataset is tiny (12 tickets ≈ $0.001)

### Vector Similarity Confusion
- **Problem:** Distance scores seem counterintuitive
- **Solution:** Emphasize lower scores = more similar (distance, not similarity)

### Chain Syntax Confusion
- **Problem:** LangChain LCEL syntax feels unfamiliar
- **Solution:** Walk through the pipe operator step-by-step

## Hands-On Challenge Facilitation

### Challenge 1: Query Testing
- Encourage students to try edge cases
- Discuss why some queries work better than others
- Connect to embedding model limitations

### Challenge 2: Parameter Tuning
- Guide discussion on k=1 vs k=5 tradeoffs
- More context isn't always better (noise vs signal)

### Challenge 3: Prompt Engineering
- Show how different prompts create different assistant personalities
- Connect to real customer service scenarios

## Assessment Opportunities

### Formative Assessment
- Can students explain the five RAG steps?
- Do they understand why chunking matters?
- Can they predict which tickets will be retrieved for a given query?

### Practical Evaluation
- Students modify prompt templates successfully
- Students can troubleshoot retrieval quality issues
- Students understand embedding cost implications

## Extension Activities

For advanced students or extra time:
1. **Custom Data:** Have them embed their own documents
2. **Hybrid Search:** Combine vector search with keyword search
3. **Multi-Modal RAG:** Add image embeddings to the pipeline
4. **Evaluation Metrics:** Implement ROUGE or BLEU scoring

## Technical Troubleshooting

### Colab Runtime Issues
- If runtime disconnects: re-run all cells from the top
- Memory errors: restart runtime and reduce chunk size
- Import errors: verify all pip installs completed

### OpenAI API Issues
- Rate limits: add sleep between requests
- Invalid key: double-check secrets configuration
- Model access: ensure gpt-3.5-turbo access

### FAISS Issues
- Installation problems: force CPU version with `faiss-cpu`
- Index errors: recreate vectorstore from scratch

## Connection to Next Module

This lab builds the "memory" component for agents. The next module (Agentic Thinking) will show how agents use this memory to make strategic decisions. Preview that they'll combine RAG with reasoning frameworks.

---

🏗️ **Under Construction**

We are constantly working to improve our resources for instructors and students.

**Have something to contribute to this Instructor Guide?** [Let us know](https://pages.git.generalassemb.ly/modular-curriculum-all-courses/universal-resources-internal/module-feedback.html).
