from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import os 
import logfire
import hashlib 
import uuid 

logfire.configure()

class RAGPipeline:
    def __init__(self):
        self.collection_name = 'gift_of_the_magi'

        self.client = AsyncOpenAI(
            api_key=os.getenv('AICREDITS_API_KEY'),
            base_url='https://api.aicredits.in/v1'
        )

        # 1. Connect to Qdrant Cloud 
        self.vector_db_client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            timeout=60.0
        )

        # 2. Safe Boot-up check
        if not self.vector_db_client.collection_exists(self.collection_name):
            logfire.info(f'Creating new RAG collection in cloud: {self.collection_name}')
            self.vector_db_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE
                )
            )
        else:
            logfire.info(f'Connected to existing RAG collection: {self.collection_name}')

    def load_documents(self) -> str: 
        filepath = '/Users/princesingh/Downloads/rag_text.txt'

        # We wrap the file read in an event span to track latency and potential disk I/O bottlenecks
        with logfire.span('rag.load_file', path=filepath) as span:
            try: 
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content:
                    raise ValueError('File is empty or blank.')
                
                # Tag the span with the character count of the raw asset
                span.set_attribute('file_characters', len(content))
                logfire.info('file_read_success', message="Document loaded successfully.")
                return content
            
            except FileNotFoundError as e:
                logfire.error('file_not_found', error=str(e))
                raise FileNotFoundError(f'File does not exist at {filepath}.')
            except Exception as e:
                logfire.error('file_load_failed', error=str(e))
                raise
        
    def chunk_document(self, text: str) -> list[str]:
       with logfire.span('rag.chunking_process') as span:
            start_marker = '*** START OF THE PROJECT GUTENBERG EBOOK THE GIFT OF THE MAGI ***'
            end_marker = '*** END OF THE PROJECT GUTENBERG EBOOK THE GIFT OF THE MAGI ***'

            start_pos = text.find(start_marker) + len(start_marker)
            end_pos = text.find(end_marker)

            # Slicing out the core text
            filtered_text = text[start_pos:end_pos].strip()
            paragraph_split = filtered_text.split('\n\n')
            paragraphs = [p.strip() for p in paragraph_split if p.strip()]

            chunk_size = 500 
            overlap = 100
            chunks = []
            current_chunk = ''

            for para in paragraphs:
                if len(current_chunk) + len(para) <= chunk_size:
                    current_chunk += para + '\n\n'
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())

                    overlap_text = current_chunk[-overlap:]
                    space_index = overlap_text.find('\n')

                    if space_index != -1:
                        overlap_text = overlap_text[space_index + 1:]

                    if len(para) > chunk_size:
                        for i in range(0, len(para), chunk_size - overlap):
                            chunks.append(para[i:i + chunk_size].strip())
                        current_chunk = ''
                    else:
                        current_chunk = overlap_text + para + '\n\n'

            if current_chunk.strip():
                chunks.append(current_chunk.strip())

            # Filtering out small or duplicate chunks
            chunks = [chunk for chunk in chunks if chunk.strip()]
            min_size = 150
            chunks = [chunk for chunk in chunks if len(chunk) >= min_size]

            seen = set()
            unique = []
            for chunk in chunks:
                if chunk not in seen:
                    seen.add(chunk)
                    unique.append(chunk)
            chunks = unique

            # CRITICAL METRIC: Track exactly how many chunks were generated from the asset
            span.set_attribute('total_chunks_generated', len(chunks))
            logfire.info('chunking_complete', chunk_count=len(chunks))
            
            return chunks
    
    async def embedding(self, texts: list[str]) -> list[dict]:
        # Wrap the API call to monitor latency and track our OpenAI costs
        with logfire.span('rag.generate_embeddings', batch_size=len(texts)) as span:
            try:
                response = await self.client.embeddings.create(
                    model='text-embedding-3-small',
                    input=texts
                )
                
                # 1. Track exactly how many tokens this batch cost us
                tokens = response.usage.prompt_tokens
                span.set_attribute('tokens_used', tokens)
                logfire.info('embeddings_generated', tokens_spent=tokens)

                embedded_chunks = []
                
                # 2. Map the generated embeddings back to their original text
                for chunk, embedding_data in zip(texts, response.data):
                    
                    # 3. Generate a "Deterministic ID" based on the text itself
                    chunk_hash = hashlib.md5(chunk.encode('utf-8')).hexdigest()
                    chunk_uuid = str(uuid.UUID(chunk_hash))

                    embedded_chunks.append({
                        'id': chunk_uuid,
                        'text': chunk,
                        'embedding': embedding_data.embedding
                    })

                return embedded_chunks

            except Exception as e:
                # Catch rate limits (429) or timeouts from OpenAI
                logfire.error('embedding_api_failed', error=str(e))
                raise
    
    def store_embeddings(self, embedded_chunks: list[dict]):
        # Track the upload process to Qdrant Cloud
        with logfire.span('rag.store_in_qdrant', payload_size=len(embedded_chunks)) as span:
            try:
                points = []
                for chunk in embedded_chunks:
                    points.append(
                        PointStruct(
                            id=chunk['id'],
                            vector=chunk['embedding'],
                            payload={'text': chunk['text']}
                        )
                    )
                
                # 4. Upload to the cloud
                self.vector_db_client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                
                logfire.info('qdrant_upload_success', uploaded_points=len(points))
                
            except Exception as e:
                # Catch network drops between your machine and Qdrant Cloud
                logfire.error('qdrant_upload_failed', error=str(e))
                raise

    def chunk_retrieval(self, query_embedding: list[float]) -> str:
        """Searches Qdrant for chunks that semantically match the user's query."""
        
        # Wrap the retrieval in a span to monitor database latency
        with logfire.span('rag.retrieve_context') as span:
            try:
                top_k = 5
                
                # 1. Query the cloud database
                results = self.vector_db_client.query_points(
                    collection_name=self.collection_name, 
                    query=query_embedding,
                    limit=top_k
                )

                # 2. Extract just the text from the payload
                chunks = [result.payload['text'] for result in results.points]
                
                # 3. Log our success metrics
                span.set_attribute('retrieved_chunks', len(chunks))
                
                # Join the chunks with double line breaks so the LLM can easily read them
                return '\n\n'.join(chunks)
                
            except Exception as e:
                # 4. Graceful Failure
                logfire.error('rag_retrieval_failed', error=str(e))
                # If Qdrant goes down, we return an empty string. 
                # This ensures your main agent can still talk to the user, it just won't have the context.
                return ""

    async def run_ingestion(self):
        """Run this EXACTLY ONCE to parse the file and upload vectors to the cloud."""
        with logfire.span('rag.full_ingestion_pipeline'):
            logfire.info("Starting document ingestion pipeline...")
            
            document = self.load_documents()
            chunks = self.chunk_document(document)
            
            embedded_chunks = await self.embedding(chunks)
            self.store_embeddings(embedded_chunks)
            
            logfire.info("Ingestion complete. Qdrant Cloud is populated.")

if __name__ == '__main__':
    # 1. We ONLY load the .env if we are running this file directly from the terminal
    from dotenv import load_dotenv
    load_dotenv()
    
    import asyncio
    
    # 2. Instantiate and run the pipeline
    pipeline = RAGPipeline()
    asyncio.run(pipeline.run_ingestion())
        