from crewai import Agent

class Agents():

    def qa_engineer(self):
        return Agent(
            role = """You are a Quality Assurance Engineer working in a software industry and your role
            is to look into the provided software and write test cases for the functionalities""",
            goal = """Navigate to the url https://serpapi.com/users/sign_in and create test cases
            using the template mentioned here https://drive.google.com/uc?export=download&id=0ByI5-ZLwpo25eXFlcU5ZMTJsT28""",
            backstory="""Create Test cases for SerpAPI Signin page""",
            verbose=True
        )

    def lead_qa_engineer(self):

        return Agent(
            role = """You are a Lead Quality Assurance Engineer working in a software industry and your role
            is to review the test cases created by Quality Assurance Engineer""",
            goal = """Validate the Test Cases provided by Quality Assurance Engineers meet the template standards as mentioned here
             https://drive.google.com/uc?export=download&id=0ByI5-ZLwpo25eXFlcU5ZMTJsT28.
             review the document covers all the functionalities in the web page mentioned and update if any functionality
             is missing in the document.""",
            backstory="""Review Test cases for  SerpAPI Signin page""",
            verbose=True
        )