from crewai import Crew, Process
from dotenv import load_dotenv
from tasks import Tasks
from agents import Agents

load_dotenv()

 
tasks = Tasks()
agents = Agents()

qa_engineer = agents.qa_engineer()
lead_qa_engineer = agents.lead_qa_engineer()

get_functionalities = tasks.get_functionalities(qa_engineer)
generate_testcases = tasks.generate_testcases(qa_engineer)
get_functionalities_lead = tasks.get_functionalities(lead_qa_engineer)
review_testcases = tasks.review_testcases(lead_qa_engineer)
display_testcases = tasks.display_testcases(qa_engineer)
 

crew = Crew(
    agents=[qa_engineer, lead_qa_engineer],
    tasks=[
        get_functionalities,
        generate_testcases,
        get_functionalities_lead,
        review_testcases
    ],
    process=Process.sequential
)

result = crew.kickoff()
print(result)