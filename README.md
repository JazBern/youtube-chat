# youtube-chat

This project focuses on using langchain to chat with youtube videos using LLMs

## Installation

```shell
pip install youtube_transcript_api
pip install -qq -U langchain tiktoken chromadb faiss-gpu
pip install -qq -U transformers InstructorEmbedding sentence_transformers
pip install -qq -U accelerate bitsandbytes xformers einops

## Usage

```python
from model import LLM
from app import TranscriptProcessor
llm = LLM('falcon').get_pipeline()
tp = TranscriptProcessor(llm,video_id = '9QiE-M1LrZk')
answer = tp.get_answer('what are some examples of high dopamine behaviours?')

## References
- [Langchain](https://python.langchain.com/docs/get_started/introduction.html)https://python.langchain.com/docs/get_started/introduction.html)
- [QA with Langchain](https://www.kaggle.com/code/hinepo/harry-potter-question-answering-with-langchain/notebook)https://www.kaggle.com/code/hinepo/harry-potter-question-answering-with-langchain/notebook)
