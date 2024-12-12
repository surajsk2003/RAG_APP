#RAG_Chat_Bot

A Personality-Based Retrieval-Augmented Generation (RAG) Chatbot, leveraging document embeddings and advanced AI models, to provide personalized, dynamic, and context-aware responses. This repository contains the source code for the chatbot, along with its various components and functionalities.


##Features
	•	Document Storage: Upload, store, and manage documents in a local SQLite database.
	•	Embedding Search: Use SentenceTransformers to convert text into embeddings for efficient search.
	•	Customizable Responses: Generate responses based on user queries, document context, and personality traits.
	•	Dynamic Chat History: Store and export user interactions.
	•	Enhanced Features: Enable advanced embeddings and persistent storage.

##Installation
	1.	Clone the repository:

git clone https://github.com/surajsk2003/RAG_Chat_Bot.git
cd RAG_Chat_Bot


	2.	Set up the virtual environment:

python3 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate


	3.	Install dependencies:

pip install -r requirements.txt


	4.	Create a .env file and set your environment variables:

OPENAI_API_KEY=your_openai_api_key


	5.	Run the application:

streamlit run app.py

##Components and Workflow

1. Environment Setup
	•	Input: .env file.
	•	Process: Load environment variables using load_dotenv.
	•	Output: Variables like OPENAI_API_KEY become accessible in the script.

2. Database Initialization
	•	Input: SQLite database (documents.db).
	•	Process: Initialize the database and create tables if they do not exist.
	•	Output: Database is ready for storing and retrieving documents.

3. File Upload
	•	Input: File uploaded by the user (CSV or TXT).
	•	Process: Parse file content and store it in the database.
	•	Output: Document text is stored in SQLite for further processing.

4. Document Embedding
	•	Input: Fetched documents from the database.
	•	Process: Convert text into embeddings using SentenceTransformer.
	•	Output: Embedding vectors are stored in memory for similarity searches.

5. Query Handling
	•	Input: User query.
	•	Process:
	1.	Encode the query into embeddings.
	2.	Use cosine similarity to find top-k relevant documents.
	•	Output: Relevant documents and similarity scores.

6. Response Generation
	•	Input: Query, personality traits, and retrieved documents.
	•	Process:
	1.	Combine inputs into a structured prompt.
	2.	Generate a response using the Llama model.
	•	Output: AI-generated response.

7. Chat History
	•	Input: User and assistant interactions.
	•	Process: Store chat history in st.session_state and export on demand.
	•	Output: Downloadable text file containing the chat history.

##Flow Diagram (Text Representation)


[User Uploads File] → [Parse File Content] → [Store in SQLite Database]
       ↑                                  ↓
[User Queries System] ← [Fetch Documents] → [Generate Embeddings]
       ↓
[Search Retrieved Docs] → [Generate AI Response] → [Display Results]

##Future Enhancements
	•	Add more personality-based configurations for responses.
	•	Extend support for additional file formats.
	•	Integrate cloud-based database storage for scalability.

##Contributing
	1.	Fork the repository.
	2.	Create a new branch:

git checkout -b feature-name


	3.	Make your changes and commit them:

git commit -m "Add feature description"


	4.	Push to your branch and create a pull request.

##License

This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to explore the code and contribute! Let me know if you’d like to refine this further.
