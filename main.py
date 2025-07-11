from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


if __name__ == '__main__':
    # write the list of the URLS of the pages
    web_paths = [
        "https://medium.com/@arthur.lagacherie/implement-the-xlstm-paper-from-scratch-with-pytorch-3a2a4ddb4f94",
    ]

    # Use WebBaseLoader to load the content of the web pages
    loader = WebBaseLoader(
        web_paths=web_paths,
    )
    docs = loader.load() # load

    print(docs[0])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)

    len(all_splits)
    # out : 14

    embedding_model = HuggingFaceEmbeddings(
        model_name="thenlper/gte-small",
        multi_process=True,
        model_kwargs={"device": "cuda"},  # set to your device
        encode_kwargs={"normalize_embeddings": True},  # Set `True` for cosine similarity
    )

    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_model)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    retrieved_docs = retriever.invoke("What is the xLSTM architecture")
    print(retrieved_docs[0].page_content)