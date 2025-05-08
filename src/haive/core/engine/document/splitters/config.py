from enum import Enum

 "TokenTextSplitter",
    "TextSplitter",
    "Tokenizer",
    "Language",
    "RecursiveCharacterTextSplitter",
    "RecursiveJsonSplitter",
    "LatexTextSplitter",
    "PythonCodeTextSplitter",
    "KonlpyTextSplitter",
    "SpacyTextSplitter",
    "NLTKTextSplitter",
    "split_text_on_tokens",
    "SentenceTransformersTokenTextSplitter",
    "ElementType",
    "HeaderType",
    "LineType",
    "HTMLHeaderTextSplitter",
    "MarkdownHeaderTextSplitter",
    "MarkdownTextSplitter",
    "CharacterTextSplitter",
]

class DocSplitterType(Enum,str):
    NLTK='nltk'
    CHARACTER='character'
    HTML='html'
    MARKDOWN='markdown'
    JSON='json'
    LATEX='latex'
    PYTHON='python'
    KONLPY='konlpy'
    SPACY='spacy'
    SENTENCE_TRANSFORMERS='sentence_transformers'
    TOKEN='token'
    TEXT='text'

class DocSplitterEngine(InvokableEngine[DocSplitterInputSchema,DocSplitterOutputSchema]):