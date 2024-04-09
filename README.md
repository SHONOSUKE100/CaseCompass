Creating a README for a complex project like the one you've described involves breaking down the elements and explaining the purpose, setup, and usage of the application in a clear and structured manner. Given the code you've shared, here’s a draft of a README that could guide users through your application.

---

# Case Compass: A Legal Precedent Search Engine

**Case Compass** is a cutting-edge legal precedent search engine designed to assist legal professionals and enthusiasts in quickly finding relevant case summaries. Leveraging state-of-the-art NLP models and a user-friendly Streamlit interface, it offers a seamless experience for navigating through a comprehensive database of legal cases.

## Features

- **Multiple Search Algorithms**: Utilize TF-IDF, Doc2Vec, and BERT models for flexible and accurate search capabilities.
- **Automatic Summary Generation**: Generate concise summaries of legal cases using Google's Generative AI.
- **Database Management**: Store and retrieve case summaries efficiently with SQLite.
- **User Interface**: A clean and intuitive interface built with Streamlit for easy navigation and operation.

## Installation

Before running the application, ensure you have Python 3.8+ installed. Clone this repository and navigate to the project directory. Install the required dependencies with:

```bash
pip install -r requirements.txt
```

This application requires access to Google Generative AI models, so make sure to set up your Google API key and store it in a `.env` file as `GOOGLE_API_KEY=<your_api_key_here>`.

## Running the Application

To launch the application, run:

```bash
streamlit run your_script_name.py
```

Replace `your_script_name.py` with the path to the script you want to run.

## Usage

Upon launching the application, select a search engine type (TF-IDF, Doc2Vec, or BERT) and enter your search query. The application will display matching cases along with options to generate summaries. Summaries are automatically saved to the database for future reference.

## Contributing

We welcome contributions! If you have suggestions for improvements or bug fixes, please feel free to make a pull request or open an issue.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.

---


