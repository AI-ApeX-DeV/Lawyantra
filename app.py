import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

st.set_page_config(page_title='Chat with Case papers', page_icon=":books:")

# Set your OpenAI API key
os.environ["openai_api_key"] = "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"

def home_page():
    st.title("LawYantra")
    st.write("Welcome to the Home Page.")
    st.title("Online Lawyer Recommendation")
    st.write("Enter your legal question to get a response.")
  
    # Create a Streamlit input text box for user input
    global chain
    chain = None
    
    user_input = st.text_area("Enter your legal question:")
    
    if st.button("Search"):
        loader = CSVLoader(file_path='PS_2_Test_Dataset.csv')

        index_creator = VectorstoreIndexCreator()
        docsearch = index_creator.from_loaders([loader])

        chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")

    
        query = user_input  # Get user input from the text area
        text = "Recommend me the best suitable lawyers to opt for this case and some details of the lawyer."
        response = chain({"question": query + text})
        st.write(response['result'])

    df = pd.read_csv('lawyer_data.csv')

    # Title
    st.title("Lawyer Data")
    df = df.dropna(axis=1, how='all')
    
    sectors = ['Environmental Law', 'Immigration Law', 'Labor Law', 'Human Rights Law', 'Tax Law',
           'Media  Entertainment Law', 'Medical Law', 'Intellectual Property Law', 'Corporate Law',
           'Banking  Finance Law', 'Family Law', 'Civil Law', 'Consumer Protection Law', 
           'Constitutional Law', 'Real Estate Law', 'Criminal  Law']

    # Create a multiselect widget for selecting multiple sectors
    selected_sectors = st.selectbox('Select Sector ', sectors)

    # Create range sliders for filtering data
    selected_jurisdiction = st.selectbox('Select a Jurisdiction', df['jurisdiction'].unique())
    selected_language = st.selectbox('Select City', df['cityofpractice'].unique())

    # Create range sliders for experience and charges
    experience_range = st.slider('Select Experience Range', min_value=0, max_value=int(df['experience'].max()), step=1, value=(0, int(df['experience'].max())))
    charges_range = st.slider('Select Charges Range', min_value=0, max_value=int(df['charges'].max()), step=10, value=(0, int(df['charges'].max())))
    disposal_days = st.slider('Select disposal days', min_value=0, max_value=int(df['daysofdisposal'].max()), step=10, value=(0, int(df['charges'].max())))

    # Filter data based on user selections
    filtered_data = df

    if selected_jurisdiction != 'None':
        filtered_data = filtered_data[filtered_data['jurisdiction'] == selected_jurisdiction]

    if selected_language != 'None':
        filtered_data = filtered_data[filtered_data['cityofpractice'] == selected_language]
        
    if selected_sectors:
       filtered_data = df[df['sector'].str.contains(selected_sectors)]


        
    filtered_data = filtered_data[(filtered_data['experience'] >= experience_range[0]) & (filtered_data['experience'] <= experience_range[1])]
    filtered_data = filtered_data[(filtered_data['charges'] >= charges_range[0]) & (filtered_data['charges'] <= charges_range[1])]
    filtered_data = filtered_data[(filtered_data['daysofdisposal'] >= disposal_days[0]) & (filtered_data['daysofdisposal'] <= disposal_days[1])]

    pro_bono_provided = st.checkbox('Pro Bono Provided')

    if pro_bono_provided :
        filtered_data = filtered_data[filtered_data['probonoserviceprovided'] == 'yes']
        
    filtered_data = filtered_data.sort_values(by='total_score', ascending=False)

    # Display filtered data
    st.write(f"Displaying {len(filtered_data)} lawyers based on your selections:")

    # Display the filtered data as a table
    st.dataframe(filtered_data)

# Define a function for Page 1
def page_1():
    st.title("Lawyer Dashboard")
    st.write("This is Lawyer Dashboad.")
        
    # Read the CSV file
    Df = pd.read_csv('vis_lawyer_data.csv')

    # Create a column selector
    lawyer_name = st.selectbox('Select a lawyer', Df['name'])

    # Filter data for the selected lawyer
    selected_lawyer_data = Df[Df['name'] == lawyer_name]

    # Display the selected lawyer's name
    st.write(f'Lawyer name: {lawyer_name}')

    # Display the selected lawyer's data

    # Calculate and display statistics
    st.title("Lawyer Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Experience", selected_lawyer_data['experience'].iloc[0])

    with col2:
        st.metric("City of Practice", selected_lawyer_data['cityofpractice'].iloc[0])

    with col3:
        st.metric("Charges per case", selected_lawyer_data['charges'].iloc[0])

    col4, col5, col6 = st.columns(3)

    with col4:
        st.metric("Ratings", selected_lawyer_data['feedback'].iloc[0])

    with col5:
        st.metric("Work sectors", selected_lawyer_data['sector'].iloc[0])

    with col6:
        st.metric("Pro Bono Services Provided", selected_lawyer_data['probonoserviceprovided'].iloc[0])
        
        # # Create a two-column layout
    col1, col2 = st.columns(2)

    # Display feedback rating in the first column
    with col1:
        feedback_rating = selected_lawyer_data['feedback'].iloc[0]
        st.header(f"Feedback Rating for {lawyer_name}")
        fig1 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=feedback_rating,
            title={'text': 'Rating'},
            gauge={'axis': {'range': [0, 5]},
                'bar': {'color': "lightblue"},
                'steps': [
                    {'range': [0, 2], 'color': "red"},
                    {'range': [2, 4], 'color': "yellow"},
                    {'range': [4, 5], 'color': "green"}]
                }))
        fig1.update_layout(width=400, height=300)  # Adjust width and height here
        st.plotly_chart(fig1)

    # Display experience in the second column
    with col2:
        experience = selected_lawyer_data['experience'].iloc[0]
        st.header(f"Experience of {lawyer_name}")
        fig2 = go.Figure(go.Indicator(
            mode="gauge+number",
            value=experience,
            title={'text': 'Experience (in Years)'},
            gauge={'axis': {'range': [0, 40]},
                'bar': {'color': "lightblue"},
                'steps': [
                    {'range': [0, 5], 'color': "red"},
                    {'range': [5, 20], 'color': "yellow"},
                    {'range': [20, 30], 'color': "green"}]
                }))
        fig2.update_layout(width=400, height=300)  # Adjust width and height here
        st.plotly_chart(fig2)

# Define a function for Page 2
def page_2():
    st.title("Study case papers")
    st.write("Study your case papers.")
    def get_pdf_text(pdf_docs):
        text = "These are the case categories:- Corporate Law, Consumer Protection Law, Labor Law, Intellectual Property Law, Criminal Law, Tax Law,Human Rights Law, Civil Law, Family Law,Family Law, Constitutional Law, Consumer Protection Law,Constitutional Law, Criminal Law, Consumer Protection Law,Intellectual Property Law, Environmental Law, Real Estate Law"
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text


    def get_text_chunks(text):
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vectorstore(text_chunks):
        embeddings = OpenAIEmbeddings()
        # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        return vectorstore

    def get_conversation_chain(vectorstore):
        llm = ChatOpenAI()
        # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain

    def handle_userinput(user_question):
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

    load_dotenv()
    # st.set_page_config(page_title = 'Chat with Case papers', page_icon = ":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation"not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("chat with Case papers :books:")
    user_question = st.text_input("Ask a question about your document")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload you case paper pdf here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Preprocessing"):
                #get the pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                #get the chunks
                text_chunks = get_text_chunks(raw_text)
                # st.write(text_chunks)

                #get the vector store
                vectorstore = get_vectorstore(text_chunks)

                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

# Create navigation buttons
page = st.sidebar.radio("Navigation", ["Home", "Page 1", "Page 2"])

# Call the appropriate function based on the selected page
if page == "Home":
    home_page()
elif page == "Page 1":
    page_1()
elif page == "Page 2":
    page_2()