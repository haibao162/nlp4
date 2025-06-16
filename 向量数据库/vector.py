from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader

# sk-Vi9p22Kog6NZz6zNFzAxO3kngbVm3WEpycPR2UtY7NNpAx9S

# 1. 加载文档数据
loader = DirectoryLoader("./doc", glob="**/*.txt", show_progress=True)
documents = loader.load()
# print(documents[0], 'documents')
# page_content='1212' metadata={'source': 'doc/vec.txt'}


# 2. 文本分块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
chunks = text_splitter.split_documents(documents)

# 3. 生成向量嵌入
embeddings = OpenAIEmbeddings(openai_api_key="sk-Vi9p22Kog6NZz6zNFzAxO3kngbVm3WEpycPR2UtY7NNpAx9S")

# 4. 存储到向量数据库
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)