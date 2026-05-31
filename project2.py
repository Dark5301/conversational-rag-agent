import os 
import asyncio 
import logfire
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.models.openai import OpenAIChatModel
from RAG import RAGPipeline
from semantic_cache import SemanticCache

load_dotenv()
logfire.configure()

class ConversationOrchestrator:
    def __init__(self, history_limit: int = 6):
        self.history_limit = history_limit
        self.conversation_history = []

        logfire.info('Booting up Conversation Orchestrator...')

        self.cache = SemanticCache(similarity_threshold = 0.80)
        self.rag = RAGPipeline()

        self.api_client = AsyncOpenAI(
            api_key = os.getenv('AICREDITS_API_KEY'),
            base_url = 'https://api.aicredits.in/v1'
        )

        custom_model = OpenAIChatModel(
            'gpt-4o-mini',
            provider = OpenAIProvider(
                openai_client=self.api_client
            )
        )

        self.main_agent = Agent(
            model = custom_model,
            system_prompt = (
                "You are a precise literary analysis assistant.\n"
                "Guardrails:\n"
                "1. Rely ONLY on the facts mentioned in the Retrieved Context OR the Previous Conversation.\n"
                "2. Do not assume, extrapolate, or bring in outside knowledge about the story.\n"
                "3. If the answer cannot be found in the Context or the History, reply gracefully that you do not have that information."
            )
        )

        self.summarizer_agent = Agent(
            model = custom_model,
            system_prompt = (
                "You are a highly efficient conversation summarizer. "
                "Your objective is to read a raw chat history and compress it into a concise, "
                "informative summary that captures the core user intent and established context. "
                "Preserve critical facts, but strip away conversational filler."
            )
        )

    async def chat(self, user_input: str) -> str:
        with logfire.span('orchestrator.chat_turn', user_input=user_input) as span:

            await self._manage_context_window()

            # 1. Convert the user's live message into a mathematical vector 
            logfire.info('Generating embedding for live user input...')
            embed_response = await self.api_client.embeddings.create(
                model='text-embedding-3-small',
                input=[user_input]
            )
            current_embedding = embed_response.data[0].embedding

            # 2. Ask the Memory (Semantic Cache)
            logfire.info('Checking Semantic Cache...')
            is_hit, cached_answer = self.cache.search(current_embedding)

            if is_hit:
                logfire.info('Cache Hit! Bypassing LLM completely.')
                self.conversation_history.append({'role': 'user', 'content': user_input})
                self.conversation_history.append({'role': 'assistant', 'content': cached_answer})
                return cached_answer
            
            # 3. Ask the Knowledge Base (RAG)
            logfire.info('Cache Miss. Retrieving RAG context...')
            rag_context = self.rag.chunk_retrieval(current_embedding)

            # 4. Construct the prompt for the Main Agent 
            # We format our clean dictionary history into a readable script for the AI
            history_text = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in self.conversation_history])

            final_prompt = (
                f"Previous Conversation:\n{history_text if history_text else 'None'}\n\n"
                f"Retrieved Context: \n{rag_context if rag_context else 'No specific context found.'}\n\n"
                f"Current User Question: {user_input}"
            )

            # 5. Run the Main Agent 
            logfire.info('Sending context and history to Main Agent...')
            result = await self.main_agent.run(final_prompt)

            # 6. Save the new knowledge to the Cache for the next user
            logfire.info('Saving new response to Qdrant Cache...')
            self.cache.save(user_input, current_embedding, result.output)

            # 7. Update our short-term working memory 
            self.conversation_history.append({'role': 'user', 'content': user_input})
            self.conversation_history.append({'role': 'assistant', 'content': result.output})

            return result.output 
        
    async def _manage_context_window(self):
        """
        Monitors the chat history. If it exceeds the limit, it compresses the older messages into a dense summary, preserving the most recent ones.
        """

        # 1. Base Case: If we are under the limit, do nothing.
        if len(self.conversation_history) <= self.history_limit:
            return 

        with logfire.span('orchestrator.compress_history') as span:
            logfire.info(f'History reached {len(self.conversation_history)} messages. Triggering compression...')

            # 2. Split the list: Compress the old, keep the new.
            # We save the last 4 messages (2 user, 2 assistant) perfectly intact so the 
            # AI doesn't lose the immediate flow of the current thought.
            messages_to_compress = self.conversation_history[:-4]
            recent_messages = self.conversation_history[-4:]

            # 3. Format the old messages into a raw string for the summarizer agent 
            raw_text = "\n".join([f'{msg['role'].upper()}: {msg['content']}' for msg in messages_to_compress])
            compression_prompt = f'Please summarize the following conversation history:\n\n{raw_text}'

            # 4. Run the Summarizer Agent 
            summary_result = await self.summarizer_agent.run(compression_prompt)

            # 5. Rewrite the history list
            # We replace 10+ old messages with 1 single system message containing the summary,
            # then append the 4 recent messages right behind it.
            self.conversation_history = [
                {
                    'role': 'assistant',
                    'content': f'[SYSTEM MEMORY: Previous conversation summary: {summary_result.output}]'
                }
            ]
            self.conversation_history.extend(recent_messages)

            span.set_attribute('Original_length', len(messages_to_compress) + 4)
            span.set_attribute('new_length', len(self.conversation_history))
            logfire.info('Context window successfully compressed.')

async def main():
    print("\n=============================================")
    print("🧠 RAG Orchestrator Initialized (Type 'exit' to quit)")
    print("=============================================\n")
    
    # Boot up the system (it handles connecting to Qdrant automatically)
    orchestrator = ConversationOrchestrator(history_limit=6)
    
    while True:
        try:
            user_text = input("\nYou: ")
            if user_text.lower() in ['exit', 'quit']:
                print("Shutting down Orchestrator. Goodbye!")
                break
                
            if not user_text.strip():
                continue
                
            # Route the message through our entire architecture
            response = await orchestrator.chat(user_text)
            
            print(f"\nAgent: {response}")
            
        except KeyboardInterrupt:
            print("\nShutting down Orchestrator. Goodbye!")
            break
        except Exception as e:
            logfire.error("terminal_crash", error=str(e))
            print(f"\n[System Error]: {e}")

if __name__ == "__main__":
    asyncio.run(main())