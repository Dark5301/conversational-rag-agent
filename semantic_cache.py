from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import logfire
import uuid
import os 

logfire.configure()

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.90):
        self.similarity_threshold = similarity_threshold
        self.collection_name = 'conversation_cache_v3'

        # 1. Connect to Qdrant Cloud 
        self.client = QdrantClient(
            url=os.getenv('QDRANT_URL'),
            api_key=os.getenv('QDRANT_API_KEY')
        )

        # 2. Boot-up check: Does our cache database exist yet?
        if not self.client.collection_exists(self.collection_name):
            logfire.info(f'Creating new Qdrant collection: {self.collection_name}')
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,
                    distance=Distance.COSINE
                )
            )

    def search(self, current_embedding: list[float]) -> tuple[bool, str | None]:
        with logfire.span('qdrant_cache_search') as span:
            # 1. Ask Qdrant to find the closest match 
            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=current_embedding,
                limit=1
            )

            # Extract the actual list of matches
            points_list = search_results.points

            # 2. Safety check: Is the database completely empty?
            if not points_list:
                span.set_attribute('cache_hit', False)
                return False, None
            
            # 3. Inspect the top result
            best_match = points_list[0] 

            # 4. Does it meet our strict threshold?
            if best_match.score >= self.similarity_threshold:
                span.set_attribute('cache_hit', True)
                span.set_attribute('similarity_score', best_match.score)
                saved_response = best_match.payload['response']
                return True, saved_response
            
            # 5. If the score is too low, we reject it
            span.set_attribute('cache_hit', False)
            span.set_attribute('highest_miss_score', best_match.score)
            return False, None

    def save(self, user_input: str, embedding: list[float], response: str):
        """
        Saves a new question and answer pair to Qdrant Cloud.
        """
        with logfire.span('qdrant_cache_save') as span:
            # 1. Generate a unique ID for this specific memory 
            point_id = str(uuid.uuid4())

            # 2. Package the data for Qdrant 
            point = PointStruct(
                id=point_id, 
                vector=embedding,
                payload={
                    'original_question': user_input,
                    'response': response
                }
            )

            # 3. Upload it to the cloud
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )

            span.set_attribute('saved_id', point_id)
            logfire.info(f'Successfully cached new response for: {user_input[:30]}...')