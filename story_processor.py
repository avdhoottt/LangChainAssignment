import os
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import click
import pickle

@dataclass
class CharacterInfo:
    name: str
    storyTitle: str
    summary: str
    relations: List[Dict[str, str]]
    characterType: str

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "storyTitle": self.storyTitle,
            "summary": self.summary,
            "relations": self.relations,
            "characterType": self.characterType
        }

class StoryProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.index = None
        self.texts = []
        self.metadatas = []
        self.index_path = "story_index.faiss"
        self.store_path = "story_store.pkl"

    def process_stories(self, stories: List[Document]) -> None:
        print("Processing stories...")
        for story in stories:
            print(f"Processing {story.metadata['source']}...")
            chunks = self.text_splitter.split_text(story.page_content)
            self.texts.extend(chunks)
            self.metadatas.extend([{"source": story.metadata["source"]} for _ in chunks])

        print("Creating embeddings...")
        embeddings = self.model.encode(self.texts)

        print("Creating FAISS index...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))

        print("Saving index and metadata...")
        faiss.write_index(self.index, self.index_path)
        with open(self.store_path, 'wb') as f:
            pickle.dump({
                "texts": self.texts,
                "metadatas": self.metadatas
            }, f)

    def find_relevant_chunks(self, query: str, k: int = 5) -> List[str]:
        query_vector = self.model.encode([query])
        D, I = self.index.search(query_vector.astype('float32'), k)
        return [self.texts[i] for i in I[0]]

    def get_character_info(self, character_name: str) -> Optional[CharacterInfo]:
        print(f"Looking up information for {character_name}...")

        if self.index is None:
            if not os.path.exists(self.index_path) or not os.path.exists(self.store_path):
                raise ValueError("No index found. Please process stories first.")

            print("Loading existing index...")
            self.index = faiss.read_index(self.index_path)
            with open(self.store_path, 'rb') as f:
                data = pickle.load(f)
                self.texts = data["texts"]
                self.metadatas = data["metadatas"]

        queries = {
            "story_title": f"story title {character_name}",
            "summary": f"summary of {character_name}'s role",
            "relations": f"relationships of {character_name}",
            "character_type": f"character type role of {character_name}"
        }

        relevant_chunks = {}
        for aspect, query in queries.items():
            chunks = self.find_relevant_chunks(query)
            relevant_chunks[aspect] = " ".join(chunks)

        story_sources = set()
        for metadata in self.metadatas:
            source = metadata["source"]
            if character_name.lower() in open(source, 'r', encoding='utf-8').read().lower():
                story_sources.add(source)

        if not story_sources:
            return None

        story_title = next(iter(story_sources)).replace('.txt', '')
        summary = relevant_chunks["summary"][:500]

        relations = []
        relations_text = relevant_chunks["relations"]
        if relations_text:
            sentences = relations_text.split('.')
            for sentence in sentences:
                if character_name in sentence:
                    relation = {"name": "Unknown", "relation": sentence.strip()}
                    relations.append(relation)

        character_type = "Supporting Character"
        type_text = relevant_chunks["character_type"].lower()
        if "protagonist" in type_text:
            character_type = "Protagonist"
        elif "antagonist" in type_text:
            character_type = "Antagonist"
        elif "main" in type_text:
            character_type = "Main Character"

        return CharacterInfo(
            name=character_name,
            storyTitle=story_title,
            summary=summary,
            relations=relations[:3],
            characterType=character_type
        )

@click.group()
def cli():
    pass

@cli.command()
@click.argument('story_files', nargs=-1, type=click.Path(exists=True))
def compute_embeddings(story_files):
    print("Initializing story processor...")
    processor = StoryProcessor()
    stories = []

    print("Reading story files...")
    for file_path in story_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            stories.append(Document(page_content=content, metadata={"source": file_path}))

    processor.process_stories(stories)
    print("Embeddings computed and stored successfully!")

@cli.command()
@click.argument('character_name')
def get_character_info(character_name):
    print(f"Looking up information for {character_name}...")
    processor = StoryProcessor()

    try:
        result = processor.get_character_info(character_name)
        if result:
            print("Character found! Here's the information:")
            print(json.dumps(result.to_json(), indent=2))
        else:
            print(f"Character '{character_name}' not found in any story.")
    except Exception as e:
        print(f"Error processing request: {str(e)}")

if __name__ == '__main__':
    cli()
