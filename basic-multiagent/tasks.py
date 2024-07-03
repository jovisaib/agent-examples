from crewai import Task

class Tasks():
    def get_functionalities(self, agent):
        return Task(
            description="""Navigate to the URL https://serpapi.com/users/sign_in and list top 5 critical functionalities to test""",
            agent=agent,
            expected_output="""List the top 5 functionalities available in the web page mentioned."""
        )

    def generate_testcases(self, agent):
        return Task(
            description="""Create Test cases for top 5 critical functionalities available in the web page""",
            agent=agent,
            expected_output="""Create Test cases for all the functionalities and create template mentioned
            in the https://drive.google.com/uc?export=download&id=0ByI5-ZLwpo25eXFlcU5ZMTJsT28 in a csv"""
        )

    def review_testcases(self, agent):
        return Task(
            description="""Review all Test cases generated in the template
            that it covers all 5 critical functionalities available in the web page.""",
            agent=agent,
            expected_output="""Review all the test cases and make sure all the test cases are available in the template
            https://drive.google.com/uc?export=download&id=0ByI5-ZLwpo25eXFlcU5ZMTJsT28"""
        )

    def display_testcases(self, agent):
        return Task(
            description="""Display all the testcases that are created in a comma seperated value(csv) format so that the users can
            take the csv formatted output""",
            agent=agent,
            expected_output="""Display the csv file in the output section so that the users can export"""
        )