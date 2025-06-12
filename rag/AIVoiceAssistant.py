from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
import warnings
from llama_index.core import SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import together

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key
api_key = os.getenv('TOGETHER_AI_API_KEY')

# Ignore warnings
warnings.filterwarnings("ignore")

# Configure the settings with a valid Hugging Face model
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")

# Disable the LLM if not needed
Settings.llm = None

class AIVoiceAssistant:
    def __init__(self):
        self._qdrant_url = "http://localhost:6333"
        self._client = QdrantClient(url=self._qdrant_url, prefer_grpc=False)
        together.api_key = api_key
        self._index = None
        self._create_kb()
        self._create_chat_engine()

    @property
    def _prompt(self):
        return """
            You are a professional AI Assistant receptionist working in Aditya's one of the best restaurant called Adii's Khana Khazana,
            Ask questions mentioned inside square brackets which you have to ask from customer, DON'T ASK THESE QUESTIONS
            IN ONE go and keep the conversation engaging! Always ask question one by one only!

            [Ask Name and contact number, what they want to order and end the conversation with greetings!]

            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Provide concise and short answers not more than 10 words, and don't chat with yourself!
            """

    def _create_chat_engine(self):
        if self._index is None:
            raise ValueError("Knowledge base has not been created.")
        memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
        self._chat_engine = self._index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self._prompt,
        )

    def _create_kb(self):
        try:
            reader = SimpleDirectoryReader(
                input_files=[r"rag/restaurant_file.txt"]
            )
            documents = reader.load_data()
            vector_store = QdrantVectorStore(client=self._client, collection_name="kitchen_db")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            self._index = VectorStoreIndex.from_documents(
                documents, storage_context=storage_context
            )
        except Exception as e:
            print(f"An error occurred while creating the knowledge base: {e}")
            raise  # Re-raise the exception to see the full traceback

    def interact_with_llm(self, prompt):
        try:
            full_prompt = f"{self._prompt}\n\nCustomer: {prompt}\nAssistant:"
            response = together.Complete.create(
                prompt=full_prompt,
                model="mistralai/Mixtral-8x7B-Instruct-v0.1",
                max_tokens=150,
                temperature=0.7,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1.1
            )
            
            # Handle different response formats
            if hasattr(response, 'choices'):
                return response.choices[0].text
            elif isinstance(response, dict):
                if 'output' in response and 'choices' in response['output']:
                    return response['output']['choices'][0]['text']
                elif 'choices' in response:
                    return response['choices'][0]['text']
            
            return "I didn't understand that. Could you please repeat?"
            
        except Exception as e:
            print(f"Error in interact_with_llm: {str(e)}")
            return "I'm having trouble responding right now. Please try again."

# Example usage
if __name__ == "__main__":
    try:
        ai_assistant = AIVoiceAssistant()
        response = ai_assistant.interact_with_llm("What is on the menu?")
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")
