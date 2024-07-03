import datasets
from functools import partial
from loguru import logger
from utils import (
    generate_stream,
    generate_with_references,
    DEBUG,
)
import typer
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt
from datasets.utils.logging import disable_progress_bar
from time import sleep

disable_progress_bar()

console = Console()

welcome_message = """
# This is a MoA (Mixture-of-Agents) interactive demo.

This demo uses the following LLMs as reference models:
- llama3-8b-8192
- llama3-70b-8192
- mixtral-8x7b-32768
- gemma-7b-it

"""

default_reference_models = [
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma-7b-it",
]


def process_fn(
    item,
    temperature=0.7,
    max_tokens=2048,
):
    references = item.get("references", [])
    model = item["model"]
    messages = item["instruction"]

    output = generate_with_references(
        model=model,
        messages=messages,
        references=references,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if DEBUG:
        logger.info(
            f"model: {model}, instruction: {item['instruction']}, output: {output[:20]}"
        )

    print(f"\nFinished querying [bold]{model}.[/bold]")

    return {"output": output}


def main(
    model: str = "llama3-70b-8192",
    reference_models: list[str] = default_reference_models,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    rounds: int = 1,
    multi_turn=True,
):
    md = Markdown(welcome_message)
    console.print(md)
    sleep(0.75)
    console.print(
        "\n[bold]To use this demo, answer the questions below to get started [cyan](press enter to use the defaults)[/cyan][/bold]:"
    )

    data = {
        "instruction": [[] for _ in range(len(reference_models))],
        "references": [""] * len(reference_models),
        "model": [m for m in reference_models],
    }

    num_proc = len(reference_models)

    model = Prompt.ask(
        "\n1. What main model do you want to use?",
        default="llama3-70b-8192",
    )
    console.print(f"Selected {model}.", style="yellow italic")
    temperature = float(
        Prompt.ask(
            "2. What temperature do you want to use? [cyan bold](0.7) [/cyan bold]",
            default=0.7,
            show_default=True,
        )
    )
    console.print(f"Selected {temperature}.", style="yellow italic")
    max_tokens = int(
        Prompt.ask(
            "3. What max tokens do you want to use? [cyan bold](2048) [/cyan bold]",
            default=2048,
            show_default=True,
        )
    )
    console.print(f"Selected {max_tokens}.", style="yellow italic")

    while True:

        try:
            instruction = Prompt.ask(
                "\n[cyan bold]Prompt >>[/cyan bold] ",
                default="Top things to do in NYC",
                show_default=True,
            )
        except EOFError:
            break

        if instruction == "exit" or instruction == "quit":
            print("Goodbye!")
            break
        if multi_turn:
            for i in range(len(reference_models)):
                data["instruction"][i].append({"role": "user", "content": instruction})
                data["references"] = [""] * len(reference_models)
        else:
            data = {
                "instruction": [[{"role": "user", "content": instruction}]]
                * len(reference_models),
                "references": [""] * len(reference_models),
                "model": [m for m in reference_models],
            }

        eval_set = datasets.Dataset.from_dict(data)

        with console.status("[bold green]Querying all the models...") as status:
            for i_round in range(rounds):
                eval_set = eval_set.map(
                    partial(
                        process_fn,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    ),
                    batched=False,
                    num_proc=num_proc,
                )
                references = [item["output"] for item in eval_set]
                data["references"] = references
                eval_set = datasets.Dataset.from_dict(data)

        console.print(
            "[cyan bold]Aggregating results & querying the aggregate model...[/cyan bold]"
        )
        output = generate_with_references(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=data["instruction"][0],
            references=references,
            generate_fn=generate_stream,
        )

        all_output = ""
        print("\n")
        console.log(Markdown(f"## Final answer from {model}"))

        for chunk in output:
            out = chunk.choices[0].delta.content
            if out is not None:
                console.print(out, end="")
                all_output += out
        print()

        if DEBUG:
            logger.info(
                f"model: {model}, instruction: {data['instruction'][0]}, output: {all_output[:20]}"
            )
        if multi_turn:
            for i in range(len(reference_models)):
                data["instruction"][i].append(
                    {"role": "assistant", "content": all_output}
                )


if __name__ == "__main__":
    typer.run(main)
