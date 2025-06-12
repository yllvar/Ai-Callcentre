# RAG AI Voice Assistant

![deepseek_mermaid_20250612_d97af6](https://github.com/user-attachments/assets/8d5a0f75-624d-43cf-807c-86c8d3a17e2d)

## **Introduction**

A voice-enabled AI assistant for a restaurant ("Adii's Khana Khazana") that:
- Listens to customer queries via microphone.
- Uses **speech-to-text** (Whisper) and **text-to-speech** (gTTS) for conversation.
- Leverages **RAG (Retrieval-Augmented Generation)** with:
  - Local knowledge base (`restaurant_file.txt`).
  - Vector database (Qdrant) for semantic search.
- Integrates **Together AI** (LLM) for generating responses.
- Designed for natural, turn-by-turn conversations (order-taking, FAQs).

---

![deepseek_mermaid_20250612_2b0abd](https://github.com/user-attachments/assets/f25b1df4-3bb9-495c-8b9e-10194604f32a)

### **File Structure & Key Components**

#### **1. `AIVoiceAssistant.py` (Core Logic)**
- **Responsibilities**:
  - Manages RAG pipeline (Qdrant vector store + local documents).
  - Handles LLM interactions (Together AI API).
  - Maintains conversation memory (`ChatMemoryBuffer`).
- **Key Features**:
  - Loads restaurant knowledge into Qdrant.
  - Uses `sentence-transformers/all-mpnet-base-v2` for embeddings.
  - System prompt enforces concise, engaging dialogues.
  - Error handling for API failures.

#### **2. `app.py` (Main Application)**
- **Workflow**:
  1. Records audio chunks (5s intervals) via PyAudio.
  2. Detects silence to avoid empty processing.
  3. Transcribes speech using **faster-whisper** (CPU-optimized).
  4. Sends text to `AIVoiceAssistant` for LLM response.
  5. Speaks responses via `voice_service`.
- **Optimizations**:
  - Dynamic chunk processing.
  - Audio stream error recovery.

#### **3. `voice_service.py` (TTS Service)**
- **Functionality**:
  - Converts text to speech using **gTTS** (Google Text-to-Speech).
  - Plays audio via **pygame** (non-blocking with threading).
  - Auto-deletes temporary audio files.
- **Error Handling**:
  - Logs TTS failures.
  - Graceful cleanup of resources.

---

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

---
