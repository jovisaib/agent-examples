import dspy
from dspy.evaluate import Evaluate
from dspy.datasets.hotpotqa import HotPotQA
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

gpt3 = dspy.OpenAI('gpt-3.5-turbo-0125', max_tokens=1000)
colbert = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')
dspy.configure(lm=gpt3, rm=colbert)


dataset = HotPotQA(train_seed=1, train_size=200, eval_seed=2023, dev_size=300, test_size=0)
trainset = [x.with_inputs('question') for x in dataset.train[0:150]]
valset = [x.with_inputs('question') for x in dataset.train[150:200]]
devset = [x.with_inputs('question') for x in dataset.dev]


agent = dspy.ReAct("question -> answer", tools=[dspy.Retrieve(k=1)])

# Set up an evaluator on the first 300 examples of the devset.
config = dict(num_threads=8, display_progress=True, display_table=5)
evaluate = Evaluate(devset=devset, metric=dspy.evaluate.answer_exact_match, **config)

print(evaluate(agent))
