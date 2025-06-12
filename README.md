# RAG AI Voice Assistant

![deepseek_mermaid_20250612_2b0abd](https://github.com/user-attachments/assets/f25b1df4-3bb9-495c-8b9e-10194604f32a)


### **Dependencies**
| Component           | Key Libraries/Tools         |
|---------------------|----------------------------|
| **Speech Processing** | PyAudio, faster-whisper    |
| **Text-to-Speech**  | gTTS, pygame               |
| **Vector Database** | Qdrant (local)             |
| **Embeddings**      | sentence-transformers      |
| **LLM**            | Together AI (Mixtral-8x7B) |
| **Conversation**    | llama-index (RAG pipeline) |

---

## Run Qdrant on Docker

Qdrant is a vector search engine designed for similarity search, nearest neighbor search, and clustering of high-dimensional data.

### Docker Installation

To use Qdrant, pull the Docker image and run it as a container:

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### **How to Run**

1. **Requirements**:
   ```bash
   pip install pyaudio faster-whisper gTTS pygame llama-index qdrant-client together python-dotenv
   ```
2. **Start Services**:
   - Ensure Qdrant is running locally (`http://localhost:6333`).
   - Set `TOGETHER_AI_API_KEY` in `.env`.
3. **Launch**:
   ```bash
   python app.py
   ```
   
![deepseek_mermaid_20250612_d97af6](https://github.com/user-attachments/assets/8d5a0f75-624d-43cf-807c-86c8d3a17e2d)

---
