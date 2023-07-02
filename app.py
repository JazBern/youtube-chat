import json
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import JSONFormatter

from langchain.vectorstores import Chroma, FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, VectorDBQA, ConversationalRetrievalChain

from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

class TranscriptProcessor:
    def __init__(self, llm, video_id):
        self.llm = llm
        self.video_id = video_id

    def get_transcript(self,video_id):
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        formatter = JSONFormatter()
        json_text = formatter.format_transcript(transcript)
        data = json.loads(json_text)
        merged_text = {}

        for item in data:
            start_time = int(item['start'])
            interval = start_time // 60
            text = item['text']

            if interval not in merged_text:
                merged_text[interval] = text
            else:
                merged_text[interval] += f" {text}"
        return merged_text

    def get_vectorstore(self,text_chunks):
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore

    def get_text_chunks(self,text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_text(text)
        return chunks
    
    def get_answer(self,query):
        transcript = self.get_transcript(self.video_id)
        text = ''.join(list(transcript.values()))
        chunks = self.get_text_chunks(text)
        db = self.get_vectorstore(chunks)
        retriever = db.as_retriever()
        qm = ConversationalRetrievalChain.from_llm(self.llm, retriever, memory=memory)
        result = qm({"question": query})
        return result['answer']
    








