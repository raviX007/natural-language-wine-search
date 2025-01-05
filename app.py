import streamlit as st
import os
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import Qdrant
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from qdrant_client import QdrantClient, models
from langchain_community.query_constructors.qdrant import QdrantTranslator



st.set_page_config(page_title="Wine Query System", layout="wide")

st.sidebar.title("Configuration")

openai_api_key = st.sidebar.text_input("Enter your OpenAI API key", type="password")


collection_name = "wine_collection"

docs = [
    Document(
        page_content="Complex, layered, rich red with dark fruit flavors",
        metadata={"name":"Opus One", "year": 2018, "rating": 96, "grape": "Cabernet Sauvignon", "color":"red", "country":"USA"},
    ),
    Document(
        page_content="Luxurious, sweet wine with flavors of honey, apricot, and peach",
        metadata={"name":"Château d'Yquem", "year": 2015, "rating": 98, "grape": "Sémillon", "color":"white", "country":"France"},
    ),
    Document(
        page_content="Full-bodied red with notes of black fruit and spice",
        metadata={"name":"Penfolds Grange", "year": 2017, "rating": 97, "grape": "Shiraz", "color":"red", "country":"Australia"},
    ),
    Document(
        page_content="Elegant, balanced red with herbal and berry nuances",
        metadata={"name":"Sassicaia", "year": 2016, "rating": 95, "grape": "Cabernet Franc", "color":"red", "country":"Italy"},
    ),
    Document(
        page_content="Highly sought-after Pinot Noir with red fruit and earthy notes",
        metadata={"name":"Domaine de la Romanée-Conti", "year": 2018, "rating": 100, "grape": "Pinot Noir", "color":"red", "country":"France"},
    ),
    Document(
        page_content="Crisp white with tropical fruit and citrus flavors",
        metadata={"name":"Cloudy Bay", "year": 2021, "rating": 92, "grape": "Sauvignon Blanc", "color":"white", "country":"New Zealand"},
    ),
    Document(
        page_content="Rich, complex Champagne with notes of brioche and citrus",
        metadata={"name":"Krug Grande Cuvée", "year": 2010, "rating": 93, "grape": "Chardonnay blend", "color":"sparkling", "country":"France"},
    ),
    Document(
        page_content="Intense, dark fruit flavors with hints of chocolate",
        metadata={"name":"Caymus Special Selection", "year": 2018, "rating": 96, "grape": "Cabernet Sauvignon", "color":"red", "country":"USA"},
    ),
    Document(
        page_content="Exotic, aromatic white with stone fruit and floral notes",
        metadata={"name":"Jermann Vintage Tunina", "year": 2020, "rating": 91, "grape": "Sauvignon Blanc blend", "color":"white", "country":"Italy"},
    ),
]

st.title("🍷 Natural Language Wine Search")
st.markdown("Search our wine collection using natural language queries. The system understands complex queries about wine characteristics, ratings, years, and more.")

st.sidebar.subheader("Example Queries")
example_queries = [
    "Show me all red wines",
    "What wines have a rating above 95?",
    "I want a wine that has fruity notes and has a rating above 97",
    "What wines come from Italy?",
    "What's a wine after 2015 but before 2020 that's earthy?",
    "Show me two wines that come from Australia or New Zealand",
]
st.sidebar.markdown("Try these example queries:")
for query in example_queries:
    st.sidebar.markdown(f"- {query}")

QDRANT_URL=st.secrets["QDRANT_URL"]
QDRANT_API_KEY=st.secrets["QDRANT_API_KEY"]

def init_qdrant():
    """Initialize Qdrant client and create collection if it doesn't exist"""
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    try:
        client.get_collection(collection_name)
    except:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=1536,  # OpenAI embedding size
                distance=models.Distance.COSINE
            ),
            
        )
    return client

if openai_api_key:
    try:
        os.environ["OPENAI_API_KEY"] = openai_api_key
        
        embeddings = OpenAIEmbeddings()
        qdrant_client = init_qdrant()
        
        vectorstore = Qdrant(
            client=qdrant_client,
            collection_name=collection_name,
            embeddings=embeddings,
            
            
        )
        
        if st.sidebar.button("Reset Database"):
            # Delete and recreate collection
            qdrant_client.delete_collection(collection_name)
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=1536,
                    distance=models.Distance.COSINE,
                    
                )
            )
            vectorstore.add_documents(docs)
            st.success("Database has been reset!")

        collection_info = qdrant_client.get_collection(collection_name)
        if collection_info.points_count == 0:
            vectorstore.add_documents(docs)
        
        llm = OpenAI(temperature=0)

        metadata_field_info = [
            AttributeInfo(
                name="grape",
                description="The grape used to make the wine",
                type="string or list[string]",
            ),
            AttributeInfo(
                name="name",
                description="The name of the wine",
                type="string or list[string]",
            ),
            AttributeInfo(
                name="color",
                description="The color of the wine",
                type="string or list[string]",
            ),
            AttributeInfo(
                name="year",
                description="The year the wine was released",
                type="integer",
            ),
            AttributeInfo(
                name="country",
                description="The name of the country the wine comes from",
                type="string",
            ),
            AttributeInfo(
                name="rating",
                description="The Robert Parker rating for the wine 0-100",
                type="integer"
            ),
        ]
        
        document_content_description = "Brief description of the wine"

        retriever = SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vectorstore,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        structured_query_translator=QdrantTranslator(metadata_key="metadata"),
        enable_limit=True,
        verbose=True,
    )
        

        query = st.text_input(
            "Enter your wine search query:",
            placeholder="e.g., Show me red wines with a rating above 95"
        )

        if query:
            with st.spinner("Searching..."):
                try:
                    results = retriever.get_relevant_documents(query)
                    
                    if results:
                        st.subheader("Search Results")
                        
                        for doc in results:
                            with st.expander(f"🍷 {doc.metadata['name']} ({doc.metadata['year']})"):
                                st.markdown(f"""
                                - **Description:** {doc.page_content}
                                - **Grape:** {doc.metadata['grape']}
                                - **Color:** {doc.metadata['color']}
                                - **Country:** {doc.metadata['country']}
                                - **Rating:** {doc.metadata['rating']}/100
                                """)
                    else:
                        st.info("No matching wines found. Try modifying your search query.")

                except Exception as search_error:
                    st.error(f"Search error: {str(search_error)}")
                    st.info("Try rephrasing your query or using simpler search terms.")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your OpenAI API key and try again.")

else:
    st.info("👈 Please enter your OpenAI API key in the sidebar to start searching.")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using LangChain and Qdrant")
