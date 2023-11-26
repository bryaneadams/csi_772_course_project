import textwrap
import re
import numpy as np
import pandas as pd
import google.generativeai as palm
from langchain.text_splitter import RecursiveCharacterTextSplitter

import langchain.schema
from typing import List, Callable, Union

from google.api_core import retry

import ast


def remove_req_qual_str(text: str) -> str:
    req_qual_str_to_remove = [
        "Who May Apply: Only applicants who meet one of the employment authority categories below are eligible to apply for this job.",
        "You will be asked to identify which category or categories you meet, and to provide documents which prove you meet the category or categories you selected. See Proof of Eligibility for an extensive list of document requirements for all employment authorities.",
        "30 Percent or More Disabled Veterans",
        "Current Department of Army Civilian Employees",
        "Current Permanent Department of Defense (DOD) Civilian Employee (non-Army)",
        "Domestic Defense Industrial Base/Major Range and Test Facilities Base Civilian Personnel Workforce",
        "Interagency Career Transition Assistance Plan",
        "Land Management Workforce Flexibility Act",
        "Non-Department of Defense (DoD) Transfer",
        "Office of Personnel Management (OPM) Interchange Agreement Eligible",
        "Priority Placement Program, DoD Military Spouse Preference (MSP) Eligible",
        "ReinstatementVeterans Employment Opportunity Act (VEOA) of 1998",
        "Additional information about transcripts is in this document.",
        "30 Percent or More Disabled Veterans",
        "Current Department of Army Civilian Employees",
        "Current Permanent Department of Defense (DOD) Civilian Employee (non-Army)",
        "Executive Order (E.O.) 12721",
        "Interagency Career Transition Assistance Plan",
        "Military Spouses, under Executive Order (E.O.) 13473",
        "Non-Appropriated Fund Instrumentality (NAFI)",
        "Non-Department of Defense (DoD) Transfer",
        "Priority Placement Program (PPP), Program S (Military Spouse) registrant",
        "Reinstatement",
        "Veterans Employment Opportunity Act (VEOA) of 1998",
        "Domestic Defense Industrial Base/Major Range and Test Facilities Base Civilian Personnel Workforce",
        "Land Management Workforce Flexibility Act",
        "Office of Personnel Management (OPM) Interchange Agreement Eligible",
        "Priority Placement Program, DoD Military Spouse Preference (MSP) Eligible",
        "Current Department of Army Civilian Employees Applying to OCONUS Positions",
        "Family Member Preference (FMP) for Overseas Employment",
        "Military Spouse Preference (MSP) for Overseas Employment",
        "Current Department of Defense (DOD) Civilian Employee (non-Army)",
        "People with Disabilities, Schedule A",
        "Interagency Career Transition Assistance Plan (ICTAP) Eligible",
        "All U.S. Citizens and Nationals with allegiance to the United States",
        "Veterans and Preference Eligible under Veterans Employment Opportunity Act (VEOA) of 1998",
        "Current Civilian Employees of the Organization",
        "Current Army Defense Civilian Intelligence Personnel System (DCIPS) Employee",
        "Current DoD Defense Civilian Intelligence Personnel System (DCIPS) Employee (non-Army)",
        "Defense Civilian Intelligence Personnel System (DCIPS) Interchange Agreement",
        "Excepted Service Overseas Family Member Appointment",
        "Land Management Workforce Flexibility Act Eligible",
        "10-Point Other Veterans? Rating",
        "5-Point Veterans' Preference",
        "Disabled Veteran w/ a Service-Connected Disability, More than 10%, Less than 30%",
        "Prior Federal Service Employee",
        "United States Citizen Applying to a DCIPS Position",
        "Current Defense Contract Management Agency Employee (DCMA)",
        "Veterans Recruitment Appointment (VRA)",
        "Proposal Evaluation (Contracting by Negotiation)",
        "External Recruitment Military Spouse Preference",
        "Current Defense Contract Management Agency (DCMA) Employee",
        "Who May Apply: US Citizen",
    ]

    for string in req_qual_str_to_remove:
        text = text.replace(string, "")

    return text


# "In order to qualify, you must meet the education and/or experience requirements described below.",
# "In order to qualify, you must meet the experience requirements described below.",
# "Experience refers to paid and unpaid experience, including volunteer work done through National Service programs (e.g., Peace Corps, AmeriCorps) and other organizations (e.g., professional; philanthropic; religious; spiritual; community; student; social). You will receive credit for all qualifying experience, including volunteer experience. Your resume must clearly describe your relevant experience; if qualifying based on education, your transcripts will be required as part of your application. ",


def remove_html_tags(text: str) -> str:
    """Text a string of html text and returns the text without the html tags

    Args:
        text (str): html string you want the tags removed from

    Returns:
        str: text with all html tags removed from it
    """
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def job_categories_to_string(value: str) -> str:
    """Transforms the job categories to just the actual occupational series

    Args:
        value (str): The value from the 'job_categories'

    Returns:
        str: string that it returns with the OSCs in one comma separated string

    Examples:
        >>> job_categories_to_string("[{'series': '0808'}, {'series': '0810'}]")
        '0808, 0810'
    """
    series_list = ast.literal_eval(value)
    series_values = [item["series"] for item in series_list]
    try:
        return ", ".join(series_values)
    except:
        return None


class VectorEmbeddings:
    def __init__(self, api_key: str):
        palm.configure(api_key=api_key)

    # @retry.Retry(timeout=30.0)
    def embed_fn(
        self, text: str, model: str = "models/embedding-gecko-001"
    ) -> Union[List[float], None]:
        """Takes a character string and returns a list of 768 floats
        This is the vector embedding size 768

        Args:
            text (str): string of text
            model (str, optional) Name of the model you want to use. Defaults to 'models/embedding-gecko-001'

        Returns:
            Union[List[float], None]: vector embeding for text string
        """
        try:
            vector = palm.generate_embeddings(model=model, text=text)["embedding"]
            return vector
        except:
            print("failed {}".format(text))
            return None

    def split_text(
        self,
        text: Union[str, list],
        chunk_size: int = 100,
        chunk_overlap: int = 20,
        length_function: Callable[[str], int] = len,
        **kwargs
    ) -> List[langchain.schema.document.Document]:
        """
        Generates chunks of text with specified size and overlap.

        Args:
            text (Union[str,list]): Maximum size of chunks to return. The funtion attempts to keep sentences together the best it can.
            chunk_size (int, optional): Overlap in characters between chunks. Defaults to 100.
            chunk_overlap (int, optional): Function that measures the length of given chunks. Defaults to 20.
            length_function (Callable[[str], int], optional): Function that measures the length of given chunks. Defaults to len.
            kwargs:  https://api.python.langchain.com/en/latest/_modules/langchain/text_splitter.html#TextSplitter

        Returns:
            List[langchain.schema.document.Document]: List of langchain document chunks
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            **kwargs
        )
        # Split the long unstructured string
        if isinstance(text, str):
            text = [text]
        chunks = text_splitter.create_documents(text)

        return chunks

    def find_top_n(
        self,
        query: str,
        df: pd.DataFrame,
        text_col: str,
        embed_col_name: str = "embeddings",
        n: int = 5,
        model: str = "models/embedding-gecko-001",
    ) -> pd.DataFrame:
        """
        Compute the distances between the query and each document in the dataframe
        using the dot product.

        Args:
            query (str): query string
            dataframe (pd.DataFrame): Dataframe for your query
            text_col (str): Name of the text column in your dataframe.
            embed_col_name (str, optional): Name of the column in your dataframe that has the vector embeddings. Defaults to 'embeddings'.
            n (int, optional): Number of observations you want to return. Defaults to 5.
            model (str, optional) Name of the model you want to use. Defaults to 'models/embedding-gecko-001'


        Returns:
            pd.DataFrame: _description_
        """
        query_embedding = palm.generate_embeddings(model=model, text=query)
        dot_products = np.dot(
            np.stack(df[embed_col_name]), query_embedding["embedding"]
        )

        score_df = pd.DataFrame(dot_products).sort_values(0, ascending=False)
        top_n = score_df[:n].index
        score = []
        text = []
        for idx in top_n:
            score.append(score_df.loc[idx][0])
            text.append(df.iloc[idx][text_col])

        return pd.DataFrame({"score": score, "text": text})

    def score_passages(
        self,
        query: str,
        df: pd.DataFrame,
        embed_col_name: str = "embeddings",
        model: str = "models/embedding-gecko-001",
    ):
        """
        Compute the distances between the query and each document in the dataframe
        using the dot product.

        Args:
            query (str): query string
            dataframe (pd.DataFrame): Dataframe for your query
            embed_col_name (str, optional): Name of the column in your dataframe that has the vector embeddings. Defaults to 'embeddings'.
            model (str, optional) Name of the model you want to use. Defaults to 'models/embedding-gecko-001'

        Returns:
            _type_: _description_
        """
        query_embedding = palm.generate_embeddings(model=model, text=query)
        dot_products = np.dot(
            np.stack(df[embed_col_name]), query_embedding["embedding"]
        )

        return dot_products

    def make_prompt(
        self, leading_text: str, query: str, relevant_passage: Union[List[str], str]
    ) -> str:
        """_summary_

        Args:
            leading_text (str): _description_
            query (str): _description_
            relevant_passage (Union[List[str],str]): _description_

        Returns:
            str: _description_
        """
        if isinstance(relevant_passage, list):
            relevant_passage = " ".join(relevant_passage)
        escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
        prompt = textwrap.dedent(
            """
        '{leading_text}'
        QUESTION: '{query}'
        PASSAGE: '{relevant_passage}'

            ANSWER:
        """
        ).format(leading_text=leading_text, query=query, relevant_passage=escaped)

        return prompt

    def generate_text(
        self,
        prompt,
        text_model,
        candidate_count: int = 3,
        temperature: float = 0.5,
        max_output_tokens: int = 1000,
    ):
        """_summary_

        Args:
            prompt (_type_): _description_
            text_model (_type_): _description_
            candidate_count (int, optional): _description_. Defaults to 3.
            temperature (float, optional): _description_. Defaults to 0.5.
            max_output_tokens (int, optional): _description_. Defaults to 1000.

        Returns:
            _type_: _description_
        """

        temperature = 0.5
        answer = palm.generate_text(
            prompt=prompt,
            model=text_model,
            candidate_count=candidate_count,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )

        return answer
